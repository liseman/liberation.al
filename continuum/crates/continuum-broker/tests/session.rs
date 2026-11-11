//! A real broker on a temporary socket, driven by a minimal client: the session handshake, heartbeats, errors, and
//! how the socket file is claimed and released.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use continuum_core::Interval;

use continuum_broker::protocol::{SCHEMA_MAJOR, SCHEMA_MINOR, read_envelope};
use continuum_broker::server::{BindError, Config, Server, lock_path};
use continuum_core::schema::continuum::{
    ControlEnvelope, ControlEnvelopeArgs, ControlPayload, ErrorCode, Goodbye, GoodbyeArgs, Heartbeat, HeartbeatArgs,
    Hello, HelloArgs, IngestNotice, IngestNoticeArgs, Purge, PurgeArgs, Query, QueryArgs, RegisterStream,
    RegisterStreamArgs, SchemaVersion, SchemaVersionArgs, StreamKind,
};
use continuum_wire::{read_frame, write_frame};
use flatbuffers::{FlatBufferBuilder, UnionWIPOffset, WIPOffset};
use tokio::net::UnixStream;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

/// A config with a small Tier 0 arena, so tests don't each map the profile's 10 GB.
fn test_config(socket: PathBuf, timeout: Duration) -> Config {
    let mut c = Config::new(socket, timeout);
    c.tier0 = Some(small_tier0());
    // Room for two small rings (the tests' 0.5 s of 10 ms PCM blocks take 96 KiB each).
    c.tier0_ring_bytes = 256 << 10;
    c
}

fn small_tier0() -> Vec<continuum_sched::SlotClass> {
    vec![
        continuum_sched::SlotClass { size_bytes: 16_384, count: 4 },
        continuum_sched::SlotClass { size_bytes: 65_536, count: 2 },
    ]
}

struct Broker {
    _dir: tempfile::TempDir,
    socket: PathBuf,
    core: continuum_broker::core::CoreHandle,
    stop: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<()>>,
}

impl Broker {
    async fn start(timeout: Duration) -> Self {
        Self::start_with(timeout, continuum_broker::server::DEFAULT_MAX_SESSIONS).await
    }

    async fn start_with(timeout: Duration, max_sessions: usize) -> Self {
        Self::start_on(timeout, max_sessions, Box::new(continuum_catalog::MemoryCatalog::new())).await
    }

    async fn start_on(timeout: Duration, max_sessions: usize, catalog: Box<dyn continuum_catalog::Catalog>) -> Self {
        Self::start_cfg(timeout, max_sessions, catalog, continuum_broker::core::ForgetConfig::default()).await
    }

    /// A broker whose purges wait for copy keepers as `forget` says.
    async fn start_forgetting(forget: continuum_broker::core::ForgetConfig) -> Self {
        let catalog = Box::new(continuum_catalog::MemoryCatalog::new());
        Self::start_cfg(Duration::from_secs(30), continuum_broker::server::DEFAULT_MAX_SESSIONS, catalog, forget).await
    }

    async fn start_cfg(
        timeout: Duration,
        max_sessions: usize,
        catalog: Box<dyn continuum_catalog::Catalog>,
        forget: continuum_broker::core::ForgetConfig,
    ) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("continuum").join("broker.sock");
        let mut config = test_config(socket.clone(), timeout);
        config.max_sessions = max_sessions;
        config.forget = forget;
        let server = Server::bind_with_catalog(config, catalog).await.unwrap();
        let core = server.core();
        let (stop, stopped) = oneshot::channel();
        let task = tokio::spawn(server.run_until(async {
            let _ = stopped.await;
        }));
        Self { _dir: dir, socket, core, stop: Some(stop), task: Some(task) }
    }

    /// The broker's Tier 0 arena, opened by name: (the arena, its name, its generation).
    fn arena(&self) -> (continuum_shm::Arena, String, u32) {
        use continuum_broker::tier0::{arena_name, generation_path};
        let file = std::fs::read_to_string(generation_path(&self.socket)).unwrap();
        let generation: u32 = file.lines().next().unwrap().parse().unwrap();
        let name = arena_name(&self.socket, generation);
        (continuum_shm::Arena::open(&name).unwrap(), name, generation)
    }

    /// Stops the broker and waits for it, handing back its directory so a test can look at it afterward.
    async fn stop(mut self) -> tempfile::TempDir {
        let _ = self.stop.take().unwrap().send(());
        self.task.take().unwrap().await.unwrap();
        self._dir
    }
}

fn envelope(
    request_id: u64,
    build: impl FnOnce(&mut FlatBufferBuilder<'static>) -> (ControlPayload, WIPOffset<UnionWIPOffset>),
) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let (payload_type, payload) = build(&mut b);
    let env =
        ControlEnvelope::create(&mut b, &ControlEnvelopeArgs { request_id, payload_type, payload: Some(payload) });
    b.finish(env, None);
    b.finished_data().to_vec()
}

fn hello(request_id: u64, major: u16) -> Vec<u8> {
    envelope(request_id, |b| {
        let schema = SchemaVersion::create(b, &SchemaVersionArgs { major, minor: SCHEMA_MINOR });
        let name = b.create_string("test-client");
        let h = Hello::create(
            b,
            &HelloArgs { client_name: Some(name), pid: std::process::id(), sdk_version: None, schema: Some(schema) },
        );
        (ControlPayload::Hello, h.as_union_value())
    })
}

fn heartbeat(request_id: u64) -> Vec<u8> {
    envelope(request_id, |b| {
        let h = Heartbeat::create(b, &HeartbeatArgs { now_ns: 0, telemetry: None });
        (ControlPayload::Heartbeat, h.as_union_value())
    })
}

fn goodbye() -> Vec<u8> {
    envelope(0, |b| {
        let g = Goodbye::create(b, &GoodbyeArgs { reason: None });
        (ControlPayload::Goodbye, g.as_union_value())
    })
}

fn query(request_id: u64) -> Vec<u8> {
    envelope(request_id, |b| {
        let q = Query::create(b, &QueryArgs::default());
        (ControlPayload::Query, q.as_union_value())
    })
}

struct Client {
    stream: UnixStream,
    buf: Vec<u8>,
    /// Messages a [`Client::recv_reply`] read past while waiting for its reply, handed out by [`Client::recv`] in
    /// the order they came.
    pending: std::collections::VecDeque<(u64, ControlPayload, Vec<u8>)>,
}

impl Client {
    async fn connect(path: &Path) -> Self {
        Self { stream: UnixStream::connect(path).await.unwrap(), buf: Vec::new(), pending: Default::default() }
    }

    async fn send(&mut self, bytes: &[u8]) {
        write_frame(&mut self.stream, bytes).await.unwrap();
    }

    /// The next reply, as (request id, payload type, raw bytes).
    /// The next message: one set aside by [`Client::recv_reply`] first, else the next off the socket, as
    /// (request id, payload type, raw bytes).
    async fn recv(&mut self) -> (u64, ControlPayload, Vec<u8>) {
        if let Some(m) = self.pending.pop_front() {
            return m;
        }
        self.read_next().await
    }

    /// The reply to request `id`, setting aside whatever else arrives first (pushes with id 0, replies to other
    /// requests, a writer's `Ingested` that beat a heartbeat's reply), for later [`Client::recv`] calls.
    async fn recv_reply(&mut self, id: u64) -> (u64, ControlPayload, Vec<u8>) {
        if let Some(i) = self.pending.iter().position(|m| m.0 == id) {
            return self.pending.remove(i).unwrap();
        }
        loop {
            let m = self.read_next().await;
            if m.0 == id {
                return m;
            }
            self.pending.push_back(m);
        }
    }

    async fn read_next(&mut self) -> (u64, ControlPayload, Vec<u8>) {
        tokio::time::timeout(Duration::from_secs(5), read_frame(&mut self.stream, &mut self.buf))
            .await
            .expect("a reply within 5 s")
            .expect("a frame");
        let env = read_envelope(&self.buf).unwrap();
        (env.request_id(), env.payload_type(), self.buf.clone())
    }

    async fn recv_error(&mut self) -> (u64, ErrorCode, String) {
        let (id, kind, bytes) = self.recv().await;
        assert_eq!(kind, ControlPayload::Error);
        let e = read_envelope(&bytes).unwrap().payload_as_error().unwrap();
        (id, e.code(), e.message().unwrap_or("").to_owned())
    }

    /// Whether the broker has closed the connection (reads EOF within 5 s).
    async fn closed(&mut self) -> bool {
        matches!(
            tokio::time::timeout(Duration::from_secs(5), read_frame(&mut self.stream, &mut self.buf)).await,
            Ok(Err(continuum_wire::FrameError::Closed | continuum_wire::FrameError::Io(_)))
        )
    }
}

#[tokio::test]
async fn hello_gets_a_welcome_then_heartbeats_are_answered() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;

    c.send(&hello(7, SCHEMA_MAJOR)).await;
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (7, ControlPayload::Welcome));
    let welcome = read_envelope(&bytes).unwrap().payload_as_welcome().unwrap();
    let session: [u8; 16] = welcome.session_id().unwrap().bytes().into();
    assert_eq!(session[6] >> 4, 7, "the session id is a UUIDv7");
    assert_eq!(welcome.schema().unwrap().major(), SCHEMA_MAJOR);

    c.send(&heartbeat(8)).await;
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (8, ControlPayload::Heartbeat));
    let now = read_envelope(&bytes).unwrap().payload_as_heartbeat().unwrap().now_ns();
    assert!(now > 1_700_000_000_000_000_000, "the broker's clock is UTC nanoseconds");

    c.send(&goodbye()).await;
    assert!(c.closed().await);
    broker.stop().await;
}

#[tokio::test]
async fn messages_before_hello_are_refused_and_the_session_stays_open() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;

    c.send(&heartbeat(3)).await;
    let (id, code, message) = c.recv_error().await;
    assert_eq!((id, code), (3, ErrorCode::NotAllowed));
    assert!(message.contains("Hello"), "{message}");

    c.send(&hello(4, SCHEMA_MAJOR)).await;
    assert_eq!(c.recv().await.1, ControlPayload::Welcome);
    broker.stop().await;
}

#[tokio::test]
async fn a_different_schema_major_is_refused_and_closed() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;
    c.send(&hello(1, SCHEMA_MAJOR + 1)).await;
    let (id, code, _) = c.recv_error().await;
    assert_eq!((id, code), (1, ErrorCode::IncompatibleSchema));
    assert!(c.closed().await);
    broker.stop().await;
}

#[tokio::test]
async fn a_second_hello_is_a_bad_request() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;
    c.send(&hello(1, SCHEMA_MAJOR)).await;
    c.recv().await;
    c.send(&hello(2, SCHEMA_MAJOR)).await;
    assert_eq!(c.recv_error().await.1, ErrorCode::BadRequest);
    broker.stop().await;
}

#[tokio::test]
async fn messages_not_built_yet_are_unavailable() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;
    c.send(&hello(1, SCHEMA_MAJOR)).await;
    c.recv().await;
    c.send(&unknown_message(9, ControlPayload::CoverageGapQuery.0)).await;
    let (id, code, message) = c.recv_error().await;
    assert_eq!((id, code), (9, ErrorCode::Unavailable));
    assert!(message.contains("CoverageGapQuery"), "{message}");
    // A Query or a Subscribe without a body is a bad request, not an unavailable one, now that both are built.
    c.send(&query(10)).await;
    assert_eq!(c.recv_error().await.1, ErrorCode::BadRequest);
    c.send(&unknown_message(11, ControlPayload::Subscribe.0)).await;
    assert_eq!(c.recv_error().await.1, ErrorCode::BadRequest);
    broker.stop().await;
}

#[tokio::test]
async fn a_resolve_frames_over_its_id_limit_is_a_bad_request() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;
    c.send(&hello(1, SCHEMA_MAJOR)).await;
    c.recv().await;
    let resolve = |request_id: u64, n: usize| {
        envelope(request_id, move |b| {
            let ids: Vec<continuum_core::schema::continuum::Uuid> =
                (0..n).map(|i| continuum_core::schema::continuum::Uuid([u8::try_from(i % 251).unwrap(); 16])).collect();
            let ids = b.create_vector(&ids);
            let r = continuum_core::schema::continuum::ResolveFrames::create(
                b,
                &continuum_core::schema::continuum::ResolveFramesArgs { ids: Some(ids) },
            );
            (ControlPayload::ResolveFrames, r.as_union_value())
        })
    };
    c.send(&resolve(2, 257)).await;
    let (id, code, message) = c.recv_error().await;
    assert_eq!((id, code), (2, ErrorCode::BadRequest));
    assert!(message.contains("at most 256"), "{message}");
    // 256 is within the limit, and so is none.
    for (request_id, n) in [(3, 256), (4, 0)] {
        c.send(&resolve(request_id, n)).await;
        let (id, kind, _) = c.recv_reply(request_id).await;
        assert_eq!((id, kind), (request_id, ControlPayload::FramesResolved));
    }
    broker.stop().await;
}

#[tokio::test]
async fn recording_applicability_bounds_are_checked_before_catalog_lookup() {
    use continuum_core::schema::continuum::{ListRecordingInputs, ListRecordingInputsArgs, TimeRange};
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;
    c.send(&hello(1, SCHEMA_MAJOR)).await;
    c.recv().await;
    for (request_id, stream_id, max_scan, expected) in
        [(2, 0, 1, ErrorCode::BadRequest), (3, 1, 4097, ErrorCode::BadRequest), (4, 1, 1, ErrorCode::NotFound)]
    {
        let request = envelope(request_id, |b| {
            let stream = b.create_string("missing.audio");
            let range = TimeRange::new(1, 2);
            let ask = ListRecordingInputs::create(
                b,
                &ListRecordingInputsArgs {
                    stream: Some(stream),
                    stream_id,
                    range: Some(&range),
                    max_scan,
                    ..Default::default()
                },
            );
            (ControlPayload::ListRecordingInputs, ask.as_union_value())
        });
        c.send(&request).await;
        let (id, code, _) = c.recv_error().await;
        assert_eq!((id, code), (request_id, expected));
    }
    c.send(&register_stream(5, "sample.audio", 100.0)).await;
    let (_, source_id) = stream_registered(&mut c).await;
    for (request_id, stream_id, expected) in [(6, source_id + 1, Some(ErrorCode::Conflict)), (7, source_id, None)] {
        let request = envelope(request_id, |b| {
            let stream = b.create_string("sample.audio");
            let range = TimeRange::new(1, 2);
            let ask = ListRecordingInputs::create(
                b,
                &ListRecordingInputsArgs { stream: Some(stream), stream_id, range: Some(&range), ..Default::default() },
            );
            (ControlPayload::ListRecordingInputs, ask.as_union_value())
        });
        c.send(&request).await;
        if let Some(expected) = expected {
            assert_eq!(c.recv_error().await.1, expected);
        } else {
            let (_, kind, bytes) = c.recv_reply(request_id).await;
            assert_eq!(kind, ControlPayload::RecordingInputs);
            let response = read_envelope(&bytes).unwrap().payload_as_recording_inputs().unwrap();
            assert_eq!(response.stream_id(), source_id);
            assert!(response.history_known());
            assert!(response.era_start_seq() > 0);
            assert!(response.inputs().unwrap().is_empty());
        }
    }
    broker.stop().await;
}

#[tokio::test]
async fn garbage_gets_a_bad_request_and_a_close() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;
    c.send(&[0xff; 7]).await;
    assert_eq!(c.recv_error().await.1, ErrorCode::BadRequest);
    assert!(c.closed().await);
    broker.stop().await;
}

#[tokio::test]
async fn a_quiet_session_is_closed_after_the_heartbeat_timeout() {
    let broker = Broker::start(Duration::from_millis(200)).await;
    let mut c = Client::connect(&broker.socket).await;
    c.send(&hello(1, SCHEMA_MAJOR)).await;
    c.recv().await;
    assert!(c.closed().await);
    broker.stop().await;
}

#[tokio::test]
async fn the_socket_is_private_and_removed_on_shutdown() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let socket = broker.socket.clone();
    let mode = std::fs::metadata(&socket).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);
    let dir_mode = std::fs::metadata(socket.parent().unwrap()).unwrap().permissions().mode() & 0o777;
    assert_eq!(dir_mode, 0o700);
    let _dir = broker.stop().await;
    assert!(socket.parent().unwrap().exists());
    assert!(!socket.exists(), "the socket file is removed on shutdown");
}

#[tokio::test]
async fn a_second_broker_on_the_same_socket_refuses_to_start() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let err = Server::bind(test_config(broker.socket.clone(), Duration::from_secs(30))).await.unwrap_err();
    assert!(matches!(err, BindError::AlreadyRunning(_)), "{err}");
    broker.stop().await;
}

#[tokio::test]
async fn a_stale_socket_is_replaced_and_a_regular_file_is_not() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("broker.sock");
    // A socket left by a broker that died: bound, then the listener dropped without removing the file.
    drop(std::os::unix::net::UnixListener::bind(&socket).unwrap());
    assert!(socket.exists());
    let server = Server::bind(test_config(socket.clone(), Duration::from_secs(30))).await.unwrap();
    drop(server);

    let file = dir.path().join("not-a-socket");
    std::fs::write(&file, b"keep me").unwrap();
    let err = Server::bind(test_config(file.clone(), Duration::from_secs(30))).await.unwrap_err();
    assert!(matches!(err, BindError::NotASocket(_)), "{err}");
    assert_eq!(std::fs::read(&file).unwrap(), b"keep me");
}

#[tokio::test]
async fn a_socket_path_too_long_for_macos_is_refused() {
    let long = PathBuf::from(format!("/tmp/{}/broker.sock", "x".repeat(100)));
    let err = Server::bind(test_config(long, Duration::from_secs(30))).await.unwrap_err();
    assert!(matches!(err, BindError::PathTooLong(_, _)), "{err}");
}

#[tokio::test]
async fn a_session_that_never_reads_its_replies_is_closed() {
    let broker = Broker::start(Duration::from_millis(300)).await;
    let mut c = Client::connect(&broker.socket).await;
    c.send(&hello(1, SCHEMA_MAJOR)).await;
    c.recv().await;
    // Send heartbeats and never read a reply. Once the replies fill the socket buffers, the broker's write can't
    // finish; it has to give up and close the session, and then our writes fail. Without a write timeout the broker
    // stops reading too, so these writes would block for good.
    let (_read, mut write) = c.stream.into_split();
    let frame = heartbeat(2);
    let outcome = tokio::time::timeout(Duration::from_secs(10), async {
        for _ in 0..1_000_000 {
            if write_frame(&mut write, &frame).await.is_err() {
                return true;
            }
        }
        false
    })
    .await;
    assert_eq!(outcome, Ok(true), "the broker closed the session instead of blocking on its write");
    broker.stop().await;
}

/// An envelope with this union discriminant and a Goodbye table as its body (a stand-in for a message from a newer
/// minor version that this broker doesn't know).
fn unknown_message(request_id: u64, discriminant: u8) -> Vec<u8> {
    envelope(request_id, |b| {
        let g = Goodbye::create(b, &GoodbyeArgs { reason: None });
        (ControlPayload(discriminant), g.as_union_value())
    })
}

fn empty_envelope(request_id: u64) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let env = ControlEnvelope::create(
        &mut b,
        &ControlEnvelopeArgs { request_id, payload_type: ControlPayload::NONE, payload: None },
    );
    b.finish(env, None);
    b.finished_data().to_vec()
}

#[tokio::test]
async fn goodbye_before_hello_closes_without_a_reply() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;
    c.send(&goodbye()).await;
    assert!(c.closed().await, "closed at once, with no Error first");
    broker.stop().await;
}

#[tokio::test]
async fn a_connection_past_the_session_cap_is_refused_and_a_freed_place_is_reused() {
    let broker = Broker::start_with(Duration::from_secs(30), 2).await;
    let mut a = Client::connect(&broker.socket).await;
    a.send(&hello(1, SCHEMA_MAJOR)).await;
    a.recv().await;
    let mut b = Client::connect(&broker.socket).await;
    b.send(&hello(1, SCHEMA_MAJOR)).await;
    b.recv().await;

    let mut c = Client::connect(&broker.socket).await;
    let (id, code, message) = c.recv_error().await;
    assert_eq!((id, code), (0, ErrorCode::Unavailable));
    assert!(message.contains("too many sessions"), "{message}");
    assert!(c.closed().await);

    a.send(&goodbye()).await;
    assert!(a.closed().await);
    // The broker notices a's session ending as it accepts, so give its loop a moment before connecting again.
    let mut d = None;
    for _ in 0..50 {
        let mut next = Client::connect(&broker.socket).await;
        next.send(&hello(2, SCHEMA_MAJOR)).await;
        if next.recv().await.1 == ControlPayload::Welcome {
            d = Some(next);
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(d.is_some(), "a place freed by a closed session is used again");
    broker.stop().await;
}

#[tokio::test]
async fn a_second_broker_is_refused_by_the_lock_even_when_the_socket_file_is_gone() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    // Without the lock, a second broker would see no socket, bind its own, and later unlink the first one's.
    std::fs::remove_file(&broker.socket).unwrap();
    let err = Server::bind(test_config(broker.socket.clone(), Duration::from_secs(30))).await.unwrap_err();
    assert!(matches!(err, BindError::AlreadyRunning(_)), "{err}");
    let lock = lock_path(&broker.socket);
    assert_eq!(std::fs::metadata(&lock).unwrap().permissions().mode() & 0o777, 0o600);
    let _dir = broker.stop().await;
    assert!(lock.exists(), "the lock file stays");
    // Once the first broker is gone, a new one starts.
    drop(
        Server::bind(test_config(_dir.path().join("continuum").join("broker.sock"), Duration::from_secs(30)))
            .await
            .unwrap(),
    );
}

#[tokio::test]
async fn a_client_that_never_says_hello_is_closed_after_the_quiet_timeout() {
    let broker = Broker::start(Duration::from_millis(200)).await;
    let mut c = Client::connect(&broker.socket).await;
    assert!(c.closed().await);
    broker.stop().await;
}

#[tokio::test]
async fn hello_without_a_schema_version_is_incompatible() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;
    let bytes = envelope(5, |b| {
        let h = Hello::create(b, &HelloArgs::default());
        (ControlPayload::Hello, h.as_union_value())
    });
    c.send(&bytes).await;
    assert_eq!(c.recv_error().await.1, ErrorCode::IncompatibleSchema);
    assert!(c.closed().await);
    broker.stop().await;
}

#[tokio::test]
async fn an_unknown_message_is_unavailable_and_an_empty_envelope_is_a_bad_request() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;
    c.send(&hello(1, SCHEMA_MAJOR)).await;
    c.recv().await;
    c.send(&unknown_message(6, 200)).await;
    assert_eq!(c.recv_error().await, (6, ErrorCode::Unavailable, "unknown is not implemented yet".into()));
    c.send(&empty_envelope(7)).await;
    assert_eq!(c.recv_error().await.0, 7);
    // The session is still open after both.
    c.send(&heartbeat(8)).await;
    assert_eq!(c.recv().await.1, ControlPayload::Heartbeat);
    broker.stop().await;
}

#[tokio::test]
async fn an_oversized_frame_gets_a_bad_request_and_a_close() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;
    let len = u32::try_from(continuum_wire::MAX_FRAME + 1).unwrap();
    tokio::io::AsyncWriteExt::write_all(&mut c.stream, &len.to_le_bytes()).await.unwrap();
    let (id, code, _) = c.recv_error().await;
    assert_eq!((id, code), (0, ErrorCode::BadRequest));
    assert!(c.closed().await);
    broker.stop().await;
}

#[tokio::test]
async fn an_abrupt_disconnect_ends_the_session_and_the_broker_keeps_accepting() {
    let broker = Broker::start_with(Duration::from_secs(30), 1).await;
    let mut c = Client::connect(&broker.socket).await;
    c.send(&hello(1, SCHEMA_MAJOR)).await;
    c.recv().await;
    drop(c);
    // With a cap of one, the next session only gets in once the dropped one has ended.
    let mut welcomed = false;
    for _ in 0..50 {
        let mut next = Client::connect(&broker.socket).await;
        next.send(&hello(2, SCHEMA_MAJOR)).await;
        if next.recv().await.1 == ControlPayload::Welcome {
            welcomed = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(welcomed);
    broker.stop().await;
}

#[tokio::test]
async fn shutdown_closes_open_sessions() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;
    c.send(&hello(1, SCHEMA_MAJOR)).await;
    c.recv().await;
    let _dir = broker.stop().await;
    assert!(c.closed().await);
}

/// Keeps sending `frame` without reading a reply, and says whether the broker closed the session (our writes fail)
/// within 10 s. A broker blocked on a write it can't finish would stop reading too, so our writes would block.
async fn sends_until_closed(stream: UnixStream, frame: &[u8]) -> bool {
    let (_read, mut write) = stream.into_split();
    tokio::time::timeout(Duration::from_secs(10), async {
        for _ in 0..1_000_000 {
            if write_frame(&mut write, frame).await.is_err() {
                return true;
            }
        }
        false
    })
    .await
    .unwrap_or(false)
}

#[tokio::test]
async fn a_session_whose_error_replies_go_unread_is_closed() {
    // Each Query is answered with an Error (Unavailable for now); none is read.
    let broker = Broker::start(Duration::from_millis(300)).await;
    let mut c = Client::connect(&broker.socket).await;
    c.send(&hello(1, SCHEMA_MAJOR)).await;
    c.recv().await;
    assert!(sends_until_closed(c.stream, &query(2)).await, "closed instead of stalling on each error reply");
    broker.stop().await;
}

#[tokio::test]
async fn a_session_whose_pre_hello_refusals_go_unread_is_closed() {
    // Before Hello, every Heartbeat is refused with NotAllowed; none is read.
    let broker = Broker::start(Duration::from_millis(300)).await;
    let c = Client::connect(&broker.socket).await;
    assert!(sends_until_closed(c.stream, &heartbeat(2)).await);
    broker.stop().await;
}

fn register_stream(request_id: u64, name: &str, rate_hz: f32) -> Vec<u8> {
    register_stream_kind(request_id, name, rate_hz, StreamKind::Sensor)
}

fn register_stream_kind(request_id: u64, name: &str, rate_hz: f32, kind: StreamKind) -> Vec<u8> {
    envelope(request_id, |b| {
        let name = b.create_string(name);
        let shape = b.create_string("NV12");
        let r = RegisterStream::create(
            b,
            &RegisterStreamArgs {
                name: Some(name),
                kind,
                shape_name: Some(shape),
                shape_version: 1,
                nominal_rate_hz: rate_hz,
                ..Default::default()
            },
        );
        (ControlPayload::RegisterStream, r.as_union_value())
    })
}

async fn stream_registered(c: &mut Client) -> (u64, u32) {
    let (id, kind, bytes) = c.recv().await;
    assert_eq!(kind, ControlPayload::StreamRegistered);
    (id, read_envelope(&bytes).unwrap().payload_as_stream_registered().unwrap().stream_id())
}

#[tokio::test]
async fn a_registered_stream_gets_its_id_and_is_live_while_its_producer_is_connected() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut a = Client::connect(&broker.socket).await;
    a.send(&hello(1, SCHEMA_MAJOR)).await;
    a.recv().await;
    a.send(&register_stream(2, "cam0", 30.0)).await;
    let (id, cam) = stream_registered(&mut a).await;
    assert_eq!(id, 2);
    let live = broker.core.live_streams().await.unwrap();
    assert!(live.contains(&continuum_core::StreamId(cam)));

    // The same stream again, from a second producer, is the same id; it stays live until both are gone.
    let mut b = Client::connect(&broker.socket).await;
    b.send(&hello(1, SCHEMA_MAJOR)).await;
    b.recv().await;
    b.send(&register_stream(3, "cam0", 30.0)).await;
    assert_eq!(stream_registered(&mut b).await.1, cam);

    a.send(&goodbye()).await;
    assert!(a.closed().await);
    let still = wait_for(&broker, |live| live.contains(&continuum_core::StreamId(cam))).await;
    assert!(still, "b still produces it");
    b.send(&goodbye()).await;
    assert!(b.closed().await);
    assert!(wait_for(&broker, |live| live.is_empty()).await, "no producer left, so not live");
    broker.stop().await;
}

/// Polls the core's live set until `want` holds, for up to 2 s (session ends reach the core asynchronously).
async fn wait_for(
    broker: &Broker,
    want: impl Fn(&std::collections::HashSet<continuum_core::StreamId>) -> bool,
) -> bool {
    for _ in 0..100 {
        if want(&broker.core.live_streams().await.unwrap()) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    false
}

#[tokio::test]
async fn a_stream_name_registered_with_different_info_is_a_conflict() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;
    c.send(&hello(1, SCHEMA_MAJOR)).await;
    c.recv().await;
    c.send(&register_stream(2, "cam0", 30.0)).await;
    stream_registered(&mut c).await;
    c.send(&register_stream_kind(3, "cam0", 30.0, StreamKind::Derived)).await;
    let (id, code, message) = c.recv_error().await;
    assert_eq!((id, code), (3, ErrorCode::Conflict));
    assert!(message.contains("cam0"), "{message}");
    broker.stop().await;
}

#[tokio::test]
async fn a_restarted_capture_worker_re_registers_with_a_new_rate_and_keeps_its_id() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;
    c.send(&hello(1, SCHEMA_MAJOR)).await;
    c.recv().await;
    c.send(&register_stream(2, "cam0", 30.0)).await;
    let (_, first) = stream_registered(&mut c).await;
    c.send(&register_stream(3, "cam0", 10.0)).await;
    assert_eq!(stream_registered(&mut c).await, (3, first));
    broker.stop().await;
}

#[tokio::test]
async fn a_bad_registration_is_a_bad_request_and_the_session_continues() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;
    c.send(&hello(1, SCHEMA_MAJOR)).await;
    c.recv().await;
    c.send(&register_stream(2, "", 30.0)).await;
    assert_eq!(c.recv_error().await.1, ErrorCode::BadRequest);
    c.send(&register_stream(3, "mic", f32::NAN)).await;
    assert_eq!(c.recv_error().await.1, ErrorCode::BadRequest);
    c.send(&register_stream(4, "mic", 50.0)).await;
    assert_eq!(stream_registered(&mut c).await.0, 4);
    broker.stop().await;
}

fn header(n: u8, stream: u32) -> Vec<u8> {
    let mut id = [0u8; 16];
    id[6] = 0x70;
    id[15] = n;
    let spec = continuum_core::HeaderSpec::raw(continuum_core::Uuid(id), continuum_core::StreamId(stream), 100, 100);
    continuum_core::HeaderBuf::build(&spec).unwrap().into_bytes()
}

fn ingest(request_id: u64, stream_id: u32, header_bytes: &[u8]) -> Vec<u8> {
    envelope(request_id, |b| {
        let bytes = b.create_vector(header_bytes);
        let n =
            IngestNotice::create(b, &IngestNoticeArgs { stream_id, header_bytes: Some(bytes), ..Default::default() });
        (ControlPayload::IngestNotice, n.as_union_value())
    })
}

async fn rejected(c: &mut Client) -> (u64, String) {
    let (id, reason, _) = rejected_with_code(c).await;
    (id, reason)
}

/// The next message, which must be an `IngestRejected`: (request id, reason, code).
async fn rejected_with_code(c: &mut Client) -> (u64, String, ErrorCode) {
    let (id, kind, bytes) = c.recv().await;
    assert_eq!(kind, ControlPayload::IngestRejected);
    let r = read_envelope(&bytes).unwrap().payload_as_ingest_rejected().unwrap();
    (id, r.reason().unwrap_or("").to_owned(), r.code())
}

#[tokio::test]
async fn only_the_streams_producer_may_ingest_and_a_bad_notice_is_rejected() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut producer = Client::connect(&broker.socket).await;
    producer.send(&hello(1, SCHEMA_MAJOR)).await;
    producer.recv().await;
    producer.send(&register_stream(2, "cam0", 30.0)).await;
    let (_, cam) = stream_registered(&mut producer).await;

    let mut other = Client::connect(&broker.socket).await;
    other.send(&hello(1, SCHEMA_MAJOR)).await;
    other.recv().await;
    other.send(&ingest(3, cam, &header(1, cam))).await;
    let (_, reason, code) = rejected_with_code(&mut other).await;
    assert!(reason.contains("didn't register") && code == ErrorCode::NotAllowed, "{reason} {code:?}");

    // The header says another stream than the notice.
    producer.send(&ingest(4, cam, &header(2, cam + 1))).await;
    assert!(rejected(&mut producer).await.1.contains("stream"));
    // Bytes that aren't a header.
    producer.send(&ingest(5, cam, &[1, 2, 3])).await;
    let (_, reason, code) = rejected_with_code(&mut producer).await;
    assert!(reason.contains("header_bytes") && code == ErrorCode::BadRequest, "{reason} {code:?}");
    broker.stop().await;
}

// ---- The no-stall rule (docs/CONTINUUM-BROKER.md): slow catalog work and a stuck client never delay anyone else ----

/// A long job for the writer: `steps` bounded steps, each taking `step` (a stand-in for a purge or a scrub).
fn slow_background_job(steps: u32, step: Duration) -> continuum_broker::core::BackgroundJob {
    let mut left = steps;
    Box::new(move |_catalog: &dyn continuum_catalog::Catalog, _budget: usize| {
        std::thread::sleep(step);
        left -= 1;
        left > 0
    })
}

/// A slow read for the reader pool (a stand-in for a large range scan).
fn slow_read(took: Duration) -> continuum_broker::core::ReadJob {
    Box::new(move |_catalog: &dyn continuum_catalog::Catalog| {
        std::thread::sleep(took);
    })
}

async fn producer(broker: &Broker) -> (Client, u32) {
    let mut c = Client::connect(&broker.socket).await;
    c.send(&hello(1, SCHEMA_MAJOR)).await;
    c.recv().await;
    c.send(&register_stream(2, "cam0", 30.0)).await;
    let (_, cam) = stream_registered(&mut c).await;
    (c, cam)
}

/// Waits until everything this session sent so far has been committed by the writer. The heartbeat's reply means
/// the session has handed every earlier frame to the core (it reads frames in order), and the sync goes through the
/// same channels after them.
async fn committed(broker: &Broker, c: &mut Client) {
    c.send(&heartbeat(98)).await;
    assert_eq!(c.recv_reply(98).await.1, ControlPayload::Heartbeat);
    broker.core.sync_writer().await.unwrap();
}

async fn heartbeat_latency(c: &mut Client) -> Duration {
    let t = std::time::Instant::now();
    c.send(&heartbeat(99)).await;
    assert_eq!(c.recv_reply(99).await.1, ControlPayload::Heartbeat);
    t.elapsed()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_ingest_waits_behind_at_most_one_step_of_a_long_job() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut c, cam) = producer(&broker).await;
    // Five 150 ms steps: 750 ms of writer work in all.
    broker.core.background(slow_background_job(5, Duration::from_millis(150))).await.unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;
    let t = std::time::Instant::now();
    c.send(&ingest(0, cam, &header(1, cam))).await;
    committed(&broker, &mut c).await;
    let waited = t.elapsed();
    assert!(waited < Duration::from_millis(400), "the ingest waited {waited:?}, more than one step");
    // And the index has it as soon as the writer says it's done: the commit came first, then the index.
    let covered = broker.core.covered(continuum_core::StreamId(cam), Interval::new(0, 1000).unwrap()).await.unwrap();
    assert_eq!(covered.len(), 0, "a payload with no copy isn't available");
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_index_shows_an_ingested_copy_once_the_writer_has_committed_it() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut c, cam) = producer(&broker).await;
    let bytes = envelope(3, |b| {
        let h = b.create_vector(&header(1, cam));
        let loc = continuum_core::StorageLocator::Chunk {
            tier: continuum_core::StorageTier::Tier1Warm,
            uri: "c".into(),
            offset: 0,
            length: 1,
        };
        let loc = flatbuffers_locator(b, &loc);
        let n = IngestNotice::create(
            b,
            &IngestNoticeArgs { stream_id: cam, header_bytes: Some(h), locator: Some(loc), ..Default::default() },
        );
        (ControlPayload::IngestNotice, n.as_union_value())
    });
    c.send(&bytes).await;
    committed(&broker, &mut c).await;
    let covered = broker.core.covered(continuum_core::StreamId(cam), Interval::new(0, 1000).unwrap()).await.unwrap();
    assert_eq!(covered.len(), 1);
    // A camera frame is an instant that stands for its stream until the next one: at 30 fps it covers one period,
    // here clipped to the query's end.
    assert_eq!(covered[0].interval, Interval::new(100, 1000).unwrap());
    assert_eq!(covered[0].tier, continuum_core::StorageTier::Tier1Warm);
    broker.stop().await;
}

/// A schema StorageLocator table for `l`, in the message being built.
fn flatbuffers_locator<'a>(
    b: &mut FlatBufferBuilder<'a>,
    l: &continuum_core::StorageLocator,
) -> flatbuffers::WIPOffset<continuum_core::schema::continuum::StorageLocator<'a>> {
    l.build_into(b)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn slow_reads_a_long_job_and_a_stuck_client_dont_delay_other_sessions() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut c, cam) = producer(&broker).await;
    // Fill every reader with a 600 ms read, start 600 ms of writer work, and attach a client that never reads.
    for _ in 0..continuum_broker::core::DEFAULT_READERS + 2 {
        broker.core.read(slow_read(Duration::from_millis(600))).await.unwrap();
    }
    broker.core.background(slow_background_job(4, Duration::from_millis(150))).await.unwrap();
    let stuck = Client::connect(&broker.socket).await;
    let flood = tokio::spawn(async move { sends_until_closed(stuck.stream, &heartbeat(5)).await });
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Meanwhile, this session's heartbeats come straight back, and its ingest waits behind one step at most.
    for _ in 0..5 {
        let took = heartbeat_latency(&mut c).await;
        assert!(took < Duration::from_millis(50), "a heartbeat took {took:?}");
    }
    let t = std::time::Instant::now();
    c.send(&ingest(0, cam, &header(2, cam))).await;
    committed(&broker, &mut c).await;
    assert!(t.elapsed() < Duration::from_millis(400), "the ingest took {:?}", t.elapsed());
    flood.abort();
    broker.stop().await;
}

// ---- Purge through the broker ----

fn chunk_at(tier: continuum_core::StorageTier) -> continuum_core::StorageLocator {
    continuum_core::StorageLocator::Chunk { tier, uri: "c".into(), offset: 0, length: 1 }
}

/// An IngestNotice for payload `n` on `stream` over [start, end), with a Tier 1 copy and these subjects.
fn ingest_copy(request_id: u64, stream: u32, n: u8, start: u64, end: u64, subjects: &[u8]) -> Vec<u8> {
    let mut id = [0u8; 16];
    id[6] = 0x70;
    id[15] = n;
    let mut spec =
        continuum_core::HeaderSpec::raw(continuum_core::Uuid(id), continuum_core::StreamId(stream), start, end);
    spec.subjects = subjects
        .iter()
        .map(|&s| {
            let mut e = [0u8; 16];
            e[6] = 0x70;
            e[14] = 0xEE;
            e[15] = s;
            continuum_core::Uuid(e)
        })
        .collect();
    let header = continuum_core::HeaderBuf::build(&spec).unwrap().into_bytes();
    envelope(request_id, |b| {
        let h = b.create_vector(&header);
        let loc = flatbuffers_locator(b, &chunk_at(continuum_core::StorageTier::Tier1Warm));
        let n = IngestNotice::create(
            b,
            &IngestNoticeArgs { stream_id: stream, header_bytes: Some(h), locator: Some(loc), ..Default::default() },
        );
        (ControlPayload::IngestNotice, n.as_union_value())
    })
}

fn purge_range(request_id: u64, start: u64, end: u64, streams: &[&str]) -> Vec<u8> {
    envelope(request_id, |b| {
        let names: Vec<_> = streams.iter().map(|s| b.create_string(s)).collect();
        let names = b.create_vector(&names);
        let reason = b.create_string("test");
        let range = continuum_core::schema::continuum::TimeRange::new(start, end);
        let p = Purge::create(
            b,
            &PurgeArgs {
                range: Some(&range),
                streams: Some(names),
                entity_id: None,
                reason: Some(reason),
                ids: None,
                detail: None,
                age_out: false,
                committed_before_ns: 0,
                aged_gap_ns: 0,
            },
        );
        (ControlPayload::Purge, p.as_union_value())
    })
}

fn purge_entity(request_id: u64, s: u8) -> Vec<u8> {
    let mut e = [0u8; 16];
    e[6] = 0x70;
    e[14] = 0xEE;
    e[15] = s;
    envelope(request_id, |b| {
        let id = continuum_core::schema::continuum::Uuid::new(&e);
        let reason = b.create_string("test");
        let p = Purge::create(
            b,
            &PurgeArgs {
                range: None,
                streams: None,
                entity_id: Some(&id),
                reason: Some(reason),
                ids: None,
                detail: None,
                age_out: false,
                committed_before_ns: 0,
                aged_gap_ns: 0,
            },
        );
        (ControlPayload::Purge, p.as_union_value())
    })
}

/// The next reply, which must be a PurgeResult: its removed counts, summed.
async fn purge_result(c: &mut Client) -> (u64, u64) {
    let (id, kind, bytes) = tokio::time::timeout(Duration::from_secs(5), c.recv()).await.unwrap();
    assert_eq!(kind, ControlPayload::PurgeResult, "expected a PurgeResult");
    let r = read_envelope(&bytes).unwrap().payload_as_purge_result().unwrap();
    (id, r.removed().map_or(0, |v| v.iter().map(|x| x.count()).sum()))
}

async fn covered_ns(broker: &Broker, cam: u32) -> Vec<(u64, u64)> {
    let all = Interval::new(0, 1_000_000).unwrap();
    let got = broker.core.covered(continuum_core::StreamId(cam), all).await.unwrap();
    got.iter().map(|c| (c.interval.start_ns(), c.interval.end_ns())).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_range_purge_removes_its_payloads_from_the_index_and_answers_with_counts() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut c, cam) = producer(&broker).await;
    for n in 0..10u8 {
        let t = u64::from(n) * 10;
        c.send(&ingest_copy(0, cam, n + 1, t, t + 10, &[])).await;
    }
    committed(&broker, &mut c).await;
    assert_eq!(covered_ns(&broker, cam).await, [(0, 100)]);

    c.send(&purge_range(30, 20, 50, &["cam0"])).await;
    let (id, removed) = purge_result(&mut c).await;
    assert_eq!((id, removed), (30, 3), "payloads at 20, 30, and 40");
    assert_eq!(covered_ns(&broker, cam).await, [(0, 20), (50, 100)]);
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_entity_purge_removes_only_that_entitys_payloads() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut c, cam) = producer(&broker).await;
    c.send(&ingest_copy(0, cam, 1, 0, 10, &[7])).await;
    c.send(&ingest_copy(0, cam, 2, 10, 20, &[])).await;
    c.send(&ingest_copy(0, cam, 3, 20, 30, &[7])).await;
    committed(&broker, &mut c).await;
    c.send(&purge_entity(13, 7)).await;
    assert_eq!(purge_result(&mut c).await, (13, 2));
    assert_eq!(covered_ns(&broker, cam).await, [(10, 20)]);
    broker.stop().await;
}

#[tokio::test]
async fn a_bad_purge_is_refused() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut c, _) = producer(&broker).await;
    c.send(&purge_range(5, 50, 20, &[])).await;
    assert_eq!(c.recv_error().await.1, ErrorCode::BadRequest, "an empty range");
    c.send(&envelope(6, |b| {
        let p = Purge::create(b, &PurgeArgs::default());
        (ControlPayload::Purge, p.as_union_value())
    }))
    .await;
    assert_eq!(c.recv_error().await.1, ErrorCode::BadRequest, "neither a range nor an entity");
    c.send(&purge_range(7, 0, 10, &["no-such-stream"])).await;
    let (id, code, message) = c.recv_error().await;
    assert_eq!((id, code), (7, ErrorCode::NotFound), "a client's typo, not a broker fault");
    assert!(message.contains("no-such-stream"), "{message}");
    broker.stop().await;
}

fn catalog_with_frames() -> (continuum_catalog::MemoryCatalog, continuum_core::StreamId) {
    use continuum_catalog::Catalog as _;
    let c = continuum_catalog::MemoryCatalog::new();
    let info = continuum_catalog::StreamInfo {
        name: "cam0".into(),
        kind: continuum_catalog::StreamKind::Sensor,
        layer: continuum_core::DomainLayer::Layer1Raw,
        format: continuum_core::PayloadFormat::RawBytes,
        shape_name: "NV12".into(),
        shape_version: 1,
        rate_hz: Some(30.0),
        anchor: None,
    };
    let cam = c.register_stream(info).unwrap();
    for n in 1..=4u8 {
        let mut id = [0u8; 16];
        id[6] = 0x70;
        id[15] = n;
        let t = u64::from(n) * 10;
        let spec = continuum_core::HeaderSpec::raw(continuum_core::Uuid(id), cam, t, t + 10);
        let header = continuum_core::HeaderBuf::build(&spec).unwrap();
        c.insert(vec![continuum_catalog::NewPayload {
            header,
            location: Some(chunk_at(continuum_core::StorageTier::Tier2Cold)),
            coverage: None,
        }])
        .unwrap();
    }
    (c, cam)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_index_is_loaded_from_the_catalog_before_the_broker_accepts() {
    let (catalog, cam) = catalog_with_frames();
    let broker = Broker::start_on(Duration::from_secs(30), 16, Box::new(catalog)).await;
    let got = broker.core.covered(cam, Interval::new(0, 100).unwrap()).await.unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!((got[0].interval.start_ns(), got[0].interval.end_ns()), (10, 50));
    assert_eq!(got[0].tier, continuum_core::StorageTier::Tier2Cold);
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_purge_left_pending_by_a_previous_run_carries_on_at_startup() {
    use continuum_catalog::Catalog as _;
    let (catalog, cam) = catalog_with_frames();
    let purge = continuum_core::Uuid([0x70; 16]);
    let selector = continuum_catalog::PurgeSelector::Range { span: Interval::new(20, 40).unwrap(), streams: None };
    catalog.start_purge(selector, "left over", "", purge, 1).unwrap();
    assert_eq!(catalog.pending_purges().unwrap(), [purge]);
    let broker = Broker::start_on(Duration::from_secs(30), 16, Box::new(catalog)).await;
    // The broker steps it to the end on its own; what it removed leaves the index.
    let mut done = false;
    for _ in 0..250 {
        let got = broker.core.covered(cam, Interval::new(0, 100).unwrap()).await.unwrap();
        let spans: Vec<_> = got.iter().map(|c| (c.interval.start_ns(), c.interval.end_ns())).collect();
        if spans == [(10, 20), (40, 50)] {
            done = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(done, "the pending purge finished and the index followed it");
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_session_whose_outbound_queue_overflows_is_closed_even_if_it_reads_again() {
    // The core drops a session whose queue fills. Its connection must close then, even though the client starts
    // reading again before the write timeout (otherwise it lingers with the core no longer knowing it).
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut c, cam) = producer(&broker).await;
    let mut stranger = Client::connect(&broker.socket).await;
    stranger.send(&hello(1, SCHEMA_MAJOR)).await;
    stranger.recv().await;
    // Each of these is rejected by the core (the stranger didn't register cam), and none is read for now.
    let bad = ingest(2, cam, &header(1, cam));
    for _ in 0..12_000 {
        if write_frame(&mut stranger.stream, &bad).await.is_err() {
            break;
        }
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    // Now read everything: the connection must end, not go on answering.
    let closed = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match read_frame(&mut stranger.stream, &mut stranger.buf).await {
                Ok(()) => {}
                Err(_) => return true,
            }
        }
    })
    .await
    .unwrap_or(false);
    assert!(closed, "the dropped session's connection closed");
    // The producer is unaffected.
    assert!(heartbeat_latency(&mut c).await < Duration::from_millis(50));
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_panicking_read_or_long_job_loses_only_itself() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut c, cam) = producer(&broker).await;
    // Every reader gets a panicking read; the pool must still run the next one.
    for _ in 0..continuum_broker::core::DEFAULT_READERS {
        broker.core.read(Box::new(|_c: &dyn continuum_catalog::Catalog| panic!("a bad read"))).await.unwrap();
    }
    let (ran, done) = tokio::sync::oneshot::channel();
    broker
        .core
        .read(Box::new(move |_c: &dyn continuum_catalog::Catalog| {
            let _ = ran.send(());
        }))
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), done).await.unwrap().unwrap();
    // A long job that panics is dropped; the writer still commits ingest.
    broker
        .core
        .background(Box::new(|_c: &dyn continuum_catalog::Catalog, _b: usize| panic!("a bad step")))
        .await
        .unwrap();
    c.send(&ingest(0, cam, &header(1, cam))).await;
    committed(&broker, &mut c).await;
    c.send(&ingest(4, cam, &header(1, cam))).await;
    assert!(ingested(&mut c).await.3 > 0, "the first ingest was committed after the panic");
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_read_and_background_queues_are_refused_as_busy() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut busy = false;
    for _ in 0..2_000 {
        if broker.core.read(slow_read(Duration::from_millis(300))).await == Err(continuum_broker::core::CoreError::Busy)
        {
            busy = true;
            break;
        }
    }
    assert!(busy, "the reader pool's queue filled and refused");
    let mut busy = false;
    for _ in 0..500 {
        let job = slow_background_job(1, Duration::from_millis(100));
        if broker.core.background(job).await == Err(continuum_broker::core::CoreError::Busy) {
            busy = true;
            break;
        }
    }
    assert!(busy, "the writer's background queue filled and refused");
    broker.stop().await;
}

#[tokio::test]
async fn welcome_is_read_the_way_clients_read_it_with_the_arena_fds_on_its_prefix() {
    // Clients read Welcome with read_frame_with_fds and match the fds to Welcome.arenas by index. With no arenas
    // yet, the broker lists none and attaches none.
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut stream = UnixStream::connect(&broker.socket).await.unwrap();
    write_frame(&mut stream, &hello(1, SCHEMA_MAJOR)).await.unwrap();
    let mut buf = Vec::new();
    let fds = continuum_wire::read_frame_with_fds(&mut stream, &mut buf, continuum_wire::MAX_FDS).await.unwrap();
    let env = read_envelope(&buf).unwrap();
    assert_eq!(env.payload_type(), ControlPayload::Welcome);
    let arenas = env.payload_as_welcome().unwrap().arenas().map_or(0, |a| a.len());
    assert_eq!(fds.len(), arenas, "one fd per arena, by index");
    // After the handshake, plain frames carry on as usual.
    write_frame(&mut stream, &heartbeat(2)).await.unwrap();
    read_frame(&mut stream, &mut buf).await.unwrap();
    assert_eq!(read_envelope(&buf).unwrap().payload_type(), ControlPayload::Heartbeat);
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_purge_naming_no_streams_covers_every_stream() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut c, cam) = producer(&broker).await;
    c.send(&register_stream(3, "mic", 50.0)).await;
    let (_, mic) = stream_registered(&mut c).await;
    c.send(&ingest_copy(0, cam, 1, 0, 10, &[])).await;
    c.send(&ingest_copy(0, mic, 2, 0, 10, &[])).await;
    committed(&broker, &mut c).await;
    c.send(&purge_range(12, 0, 10, &[])).await;
    assert_eq!(purge_result(&mut c).await, (12, 2));
    assert!(covered_ns(&broker, cam).await.is_empty());
    assert!(covered_ns(&broker, mic).await.is_empty());
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_purge_finishes_after_its_requester_disconnects() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut producer_c, cam) = producer(&broker).await;
    for n in 0..20u8 {
        let t = u64::from(n) * 10;
        producer_c.send(&ingest_copy(0, cam, n + 1, t, t + 10, &[])).await;
    }
    committed(&broker, &mut producer_c).await;
    let mut asker = Client::connect(&broker.socket).await;
    asker.send(&hello(1, SCHEMA_MAJOR)).await;
    asker.recv().await;
    asker.send(&purge_range(2, 0, 200, &["cam0"])).await;
    drop(asker);
    let mut emptied = false;
    for _ in 0..250 {
        if covered_ns(&broker, cam).await.is_empty() {
            emptied = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(emptied, "the purge ran to the end with nobody to answer");
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn more_pending_purges_than_the_background_queue_holds_all_carry_on() {
    use continuum_catalog::Catalog as _;
    let (catalog, cam) = catalog_with_frames();
    // 80 pending purges (the queue holds 64), each over its own empty stretch, plus one over a real payload last.
    for n in 0..80u32 {
        let mut id = [0x70u8; 16];
        id[12..].copy_from_slice(&n.to_be_bytes());
        let t = 1_000 + u64::from(n) * 10;
        let span = Interval::new(t, t + 5).unwrap();
        catalog
            .start_purge(
                continuum_catalog::PurgeSelector::Range { span, streams: None },
                "old",
                "",
                continuum_core::Uuid(id),
                1,
            )
            .unwrap();
    }
    let last = continuum_core::Uuid([0x71; 16]);
    let span = Interval::new(10, 20).unwrap();
    catalog.start_purge(continuum_catalog::PurgeSelector::Range { span, streams: None }, "old", "", last, 1).unwrap();
    assert_eq!(catalog.pending_purges().unwrap().len(), 81);
    let broker = Broker::start_on(Duration::from_secs(30), 16, Box::new(catalog)).await;
    let mut done = false;
    for _ in 0..250 {
        let got = broker.core.covered(cam, Interval::new(0, 100).unwrap()).await.unwrap();
        if got.first().map(|c| c.interval.start_ns()) == Some(20) {
            done = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(done, "the 81st pending purge ran too");
    broker.stop().await;
}

// ---- The scheduler through the broker: workers, leases, and the P0 voice turn ----

use continuum_core::schema::continuum as fbs;

fn register_derived(request_id: u64, name: &str) -> Vec<u8> {
    register_stream_kind(request_id, name, 0.0, StreamKind::Derived)
}

/// A worker with one GPU capability producing `produces` from `requires`, preemptible.
fn register_worker(request_id: u64, produces: &str, requires: &str) -> Vec<u8> {
    envelope(request_id, |b| {
        let req_name = b.create_string(requires);
        let req =
            fbs::Requirement::create(b, &fbs::RequirementArgs { stream: Some(req_name), before_ns: 0, after_ns: 0 });
        let reqs = b.create_vector(&[req]);
        let produces = b.create_string(produces);
        let cap = fbs::Capability::create(
            b,
            &fbs::CapabilityArgs {
                produces: Some(produces),
                requires: Some(reqs),
                compute_domain: continuum_core::ComputeDomain::GPU,
                ram_mb: 100,
                vram_mb: 500,
                est_ms_per_item: 5.0,
                max_batch: 8,
                preemptible: true,
                role: fbs::CapabilityRole::Produce,
                outputs_per_job: 1,
                ..Default::default()
            },
        );
        let caps = b.create_vector(&[cap]);
        let name = b.create_string("pose-worker");
        let w = fbs::RegisterWorker::create(
            b,
            &fbs::RegisterWorkerArgs {
                worker_id: Some(name),
                version_hash: None,
                capabilities: Some(caps),
                device_index: None,
                keeps_copies: false,
            },
        );
        (ControlPayload::RegisterWorker, w.as_union_value())
    })
}

fn quote_request(request_id: u64, capability_id: u32, band: continuum_core::PriorityBand) -> Vec<u8> {
    envelope(request_id, |b| {
        let range = fbs::TimeRange::new(0, 1_000);
        let q = fbs::QuoteRequest::create(
            b,
            &fbs::QuoteRequestArgs {
                band,
                compute_domain: continuum_core::ComputeDomain::GPU,
                est_duration_ms: 1_000,
                capability_id,
                range: Some(&range),
                ..Default::default()
            },
        );
        (ControlPayload::QuoteRequest, q.as_union_value())
    })
}

fn set_voice(voice: continuum_core::VoiceTurnState) -> Vec<u8> {
    envelope(0, |b| {
        let s = fbs::SetSystemState::create(b, &fbs::SetSystemStateArgs { voice_turn: voice });
        (ControlPayload::SetSystemState, s.as_union_value())
    })
}

fn release(lease_id: u64) -> Vec<u8> {
    envelope(0, |b| {
        let r = fbs::LeaseReleased::create(b, &fbs::LeaseReleasedArgs { lease_id, output_seqs: None });
        (ControlPayload::LeaseReleased, r.as_union_value())
    })
}

/// A worker session with "cam0" (sensor) and "pose" (derived) registered and its capability accepted; returns its
/// capability id.
async fn worker(broker: &Broker) -> (Client, u32, u32, u32) {
    let (mut c, cam) = producer(broker).await;
    c.send(&register_derived(3, "pose")).await;
    let (_, pose) = stream_registered(&mut c).await;
    c.send(&register_worker(4, "pose", "cam0")).await;
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (4, ControlPayload::WorkerRegistered));
    let w = read_envelope(&bytes).unwrap().payload_as_worker_registered().unwrap();
    assert_eq!(w.accepted().unwrap().iter().collect::<Vec<_>>(), [0]);
    assert!(w.runnable().unwrap().get(0), "this box's profile has a GPU");
    let cap = w.capability_ids().unwrap().get(0);
    (c, cap, pose, cam)
}

async fn grant(c: &mut Client, request_id: u64, cap: u32, band: continuum_core::PriorityBand) -> u64 {
    c.send(&quote_request(request_id, cap, band)).await;
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (request_id, ControlPayload::LeaseGrant));
    let g = read_envelope(&bytes).unwrap().payload_as_lease_grant().unwrap();
    assert!(g.granted(), "the lease was granted");
    g.lease().unwrap().lease_id()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_capability_naming_an_unknown_stream_is_refused_and_quotes_need_a_worker() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut c, _) = producer(&broker).await;
    c.send(&quote_request(3, 0, continuum_core::PriorityBand::P3Background)).await;
    assert_eq!(c.recv_error().await, (3, ErrorCode::NotAllowed, "register a worker first".into()));
    c.send(&register_worker(4, "no-such-output", "cam0")).await;
    let (_, kind, bytes) = c.recv().await;
    assert_eq!(kind, ControlPayload::WorkerRegistered);
    let w = read_envelope(&bytes).unwrap().payload_as_worker_registered().unwrap();
    assert_eq!(w.accepted().map_or(0, |a| a.len()), 0);
    assert!(w.refused_reasons().unwrap().get(0).contains("no-such-output"));
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_p0_voice_turn_preempts_a_p3_gpu_lease_within_5_ms_across_real_sockets() {
    use continuum_core::{PriorityBand, VoiceTurnState};
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut w, cap, _, _) = worker(&broker).await;
    let mut voice = Client::connect(&broker.socket).await;
    voice.send(&hello(1, SCHEMA_MAJOR)).await;
    voice.recv().await;

    // Twenty trials, every one printed (run with --nocapture to see them). The assertion is on the best, since this
    // is a shared, often loaded machine; the spread is for judging whether a slow one is the broker or the box.
    let mut took_all = Vec::new();
    for trial in 0..20u64 {
        let lease = grant(&mut w, 10 + trial, cap, PriorityBand::P3Background).await;
        let t = std::time::Instant::now();
        voice.send(&set_voice(VoiceTurnState::Listening)).await;
        // The preemption comes before the broadcast.
        let (id, kind, bytes) = w.recv().await;
        let took = t.elapsed();
        assert_eq!((id, kind), (0, ControlPayload::LeasePreempted), "pushed with request id 0");
        let p = read_envelope(&bytes).unwrap().payload_as_lease_preempted().unwrap();
        assert_eq!(p.lease_id(), lease);
        assert_eq!(p.by_band(), PriorityBand::P0Critical);
        took_all.push(took);
        // Both sessions hear the new state.
        assert_eq!(w.recv().await.1, ControlPayload::SystemStateChanged);
        let (id, kind, bytes) = voice.recv().await;
        assert_eq!((id, kind), (0, ControlPayload::SystemStateChanged));
        assert_eq!(
            read_envelope(&bytes).unwrap().payload_as_system_state_changed().unwrap().voice_turn(),
            VoiceTurnState::Listening
        );
        // Give the lease back and end the turn for the next trial.
        w.send(&release(lease)).await;
        voice.send(&set_voice(VoiceTurnState::Idle)).await;
        assert_eq!(w.recv().await.1, ControlPayload::SystemStateChanged);
        assert_eq!(voice.recv().await.1, ControlPayload::SystemStateChanged);
    }
    let mut sorted = took_all.clone();
    sorted.sort();
    eprintln!("preemption latencies (us): {:?}", took_all.iter().map(Duration::as_micros).collect::<Vec<_>>());
    eprintln!("min {:?}, median {:?}, max {:?}", sorted[0], sorted[sorted.len() / 2], sorted[sorted.len() - 1]);
    let (best, median) = (sorted[0], sorted[sorted.len() / 2]);
    assert!(best < Duration::from_millis(5), "the best preemption took {best:?}");
    assert!(median < Duration::from_millis(5), "the median preemption took {median:?}");
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_committed_lease_indexes_its_outputs() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut w, cap, pose, _) = worker(&broker).await;
    let lease = grant(&mut w, 5, cap, continuum_core::PriorityBand::P2Nearline).await;
    let mut id = [0u8; 16];
    id[6] = 0x70;
    id[15] = 42;
    let spec = continuum_core::HeaderSpec::raw(continuum_core::Uuid(id), continuum_core::StreamId(pose), 100, 200);
    let header = continuum_core::HeaderBuf::build(&spec).unwrap().into_bytes();
    let commit = envelope(6, |b| {
        let h = b.create_vector(&header);
        let loc = flatbuffers_locator(b, &chunk_at(continuum_core::StorageTier::Tier1Warm));
        let item = fbs::Item::create(b, &fbs::ItemArgs { locator: Some(loc), header_bytes: Some(h), commit_seq: 0 });
        let outputs = b.create_vector(&[item]);
        let c = fbs::CommitLease::create(
            b,
            &fbs::CommitLeaseArgs { lease_id: lease, outputs: Some(outputs), coverage: None, cost: None },
        );
        (ControlPayload::CommitLease, c.as_union_value())
    });
    w.send(&commit).await;
    acked(&mut w, 6, lease).await;
    let got = broker.core.covered(continuum_core::StreamId(pose), Interval::new(0, 1_000).unwrap()).await.unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!((got[0].interval.start_ns(), got[0].interval.end_ns()), (100, 200));
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn only_the_lease_holder_may_commit_abort_renew_or_release_it_and_only_on_its_stream() {
    use continuum_core::PriorityBand;
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut w, cap, pose, cam) = worker(&broker).await;
    let lease = grant(&mut w, 5, cap, PriorityBand::P2Nearline).await;
    let header_on = |stream: u32, n: u8| {
        let mut id = [0u8; 16];
        id[6] = 0x70;
        id[15] = n;
        let spec = continuum_core::HeaderSpec::raw(continuum_core::Uuid(id), continuum_core::StreamId(stream), 1, 2);
        continuum_core::HeaderBuf::build(&spec).unwrap().into_bytes()
    };
    let commit_of = |request_id: u64, header: Vec<u8>| {
        envelope(request_id, |b| {
            let h = b.create_vector(&header);
            let item = fbs::Item::create(b, &fbs::ItemArgs { locator: None, header_bytes: Some(h), commit_seq: 0 });
            let outputs = b.create_vector(&[item]);
            let c = fbs::CommitLease::create(
                b,
                &fbs::CommitLeaseArgs { lease_id: lease, outputs: Some(outputs), coverage: None, cost: None },
            );
            (ControlPayload::CommitLease, c.as_union_value())
        })
    };
    // Another session, even one that registered a worker of its own, can't touch the lease.
    let (mut other, _, _) = {
        let mut c = Client::connect(&broker.socket).await;
        c.send(&hello(1, SCHEMA_MAJOR)).await;
        c.recv().await;
        c.send(&register_worker(2, "pose", "cam0")).await;
        assert_eq!(c.recv().await.1, ControlPayload::WorkerRegistered);
        (c, 0, 0)
    };
    other.send(&commit_of(10, header_on(pose, 1))).await;
    assert_eq!(other.recv_error().await.1, ErrorCode::NotAllowed);
    let abort = envelope(11, |b| {
        let a = fbs::AbortLease::create(b, &fbs::AbortLeaseArgs { lease_id: lease, reason: None });
        (ControlPayload::AbortLease, a.as_union_value())
    });
    other.send(&abort).await;
    assert_eq!(other.recv_error().await.1, ErrorCode::NotAllowed);
    let renew = envelope(12, |b| {
        let r = fbs::RenewLease::create(b, &fbs::RenewLeaseArgs { lease_id: lease, extend_ms: 0 });
        (ControlPayload::RenewLease, r.as_union_value())
    });
    other.send(&renew).await;
    assert_eq!(other.recv_error().await.1, ErrorCode::NotAllowed);
    other.send(&release(lease)).await;
    // The holder's lease is untouched: it still renews, and an output on another stream is refused.
    w.send(&renew).await;
    assert_eq!(w.recv().await.1, ControlPayload::LeaseGrant);
    w.send(&commit_of(13, header_on(cam, 2))).await;
    let (id, code, message) = w.recv_error().await;
    assert_eq!((id, code), (13, ErrorCode::NotAllowed));
    assert!(message.contains("not the lease's"), "{message}");
    // On its own stream the commit goes through.
    w.send(&commit_of(14, header_on(pose, 3))).await;
    acked(&mut w, 14, lease).await;
    broker.stop().await;
}

#[tokio::test]
async fn hello_and_more_frames_in_one_write_are_all_answered_in_order() {
    // The handshake must not read past Hello: frames that arrive with it belong to the session after.
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut stream = UnixStream::connect(&broker.socket).await.unwrap();
    let mut bytes = Vec::new();
    for frame in [hello(1, SCHEMA_MAJOR), heartbeat(2), heartbeat(3)] {
        bytes.extend_from_slice(&u32::try_from(frame.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&frame);
    }
    tokio::io::AsyncWriteExt::write_all(&mut stream, &bytes).await.unwrap();
    let mut buf = Vec::new();
    let mut got = Vec::new();
    for _ in 0..3 {
        read_frame(&mut stream, &mut buf).await.unwrap();
        let env = read_envelope(&buf).unwrap();
        got.push((env.request_id(), env.payload_type()));
    }
    assert_eq!(got, [(1, ControlPayload::Welcome), (2, ControlPayload::Heartbeat), (3, ControlPayload::Heartbeat)]);
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_bad_notice_inside_a_batch_is_rejected_by_its_own_request_id_and_its_neighbors_go_in() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut c, cam) = producer(&broker).await;
    c.send(&ingest_copy(0, cam, 1, 0, 10, &[])).await;
    committed(&broker, &mut c).await;
    // Three notices sent together, so they land in one writer tick: the middle one reuses an indexed id with a
    // different header (a conflict; a repeat with the same bytes would be idempotent).
    let mut bytes = Vec::new();
    let batch = [
        ingest_copy(20, cam, 2, 10, 20, &[]),
        ingest_copy(21, cam, 1, 5, 15, &[]),
        ingest_copy(22, cam, 3, 20, 30, &[]),
    ];
    for frame in batch {
        bytes.extend_from_slice(&u32::try_from(frame.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&frame);
    }
    tokio::io::AsyncWriteExt::write_all(&mut c.stream, &bytes).await.unwrap();
    let (id, _, payload, seq2) = ingested(&mut c).await;
    assert_eq!((id, payload), (20, 2));
    let (id, _) = rejected(&mut c).await;
    assert_eq!(id, 21, "the rejection names the bad notice");
    let (id, _, payload, seq3) = ingested(&mut c).await;
    assert_eq!((id, payload), (22, 3));
    assert!(seq2 < seq3);
    committed(&broker, &mut c).await;
    // The neighbors were indexed: repeating them answers with their sequences, and the index has all three.
    assert_eq!(covered_ns(&broker, cam).await, [(0, 30)]);
    c.send(&ingest_copy(30, cam, 2, 10, 20, &[])).await;
    assert_eq!(ingested(&mut c).await, (30, cam, 2, seq2));
    c.send(&ingest_copy(31, cam, 3, 20, 30, &[])).await;
    assert_eq!(ingested(&mut c).await, (31, cam, 3, seq3));
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_hydrate_lease_may_commit_onto_the_stream_it_decodes() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut c, cam) = producer(&broker).await;
    let reg = envelope(3, |b| {
        let cap = fbs::Capability::create(
            b,
            &fbs::CapabilityArgs {
                compute_domain: continuum_core::ComputeDomain::CPU,
                est_ms_per_item: 1.0,
                max_batch: 8,
                preemptible: true,
                role: fbs::CapabilityRole::Hydrate,
                outputs_per_job: 1,
                ..Default::default()
            },
        );
        let caps = b.create_vector(&[cap]);
        let name = b.create_string("tier-worker");
        let w = fbs::RegisterWorker::create(
            b,
            &fbs::RegisterWorkerArgs {
                worker_id: Some(name),
                version_hash: None,
                capabilities: Some(caps),
                device_index: None,
                keeps_copies: false,
            },
        );
        (ControlPayload::RegisterWorker, w.as_union_value())
    });
    c.send(&reg).await;
    let (_, kind, bytes) = c.recv().await;
    assert_eq!(kind, ControlPayload::WorkerRegistered);
    let cap = read_envelope(&bytes).unwrap().payload_as_worker_registered().unwrap().capability_ids().unwrap().get(0);
    let quote = envelope(4, |b| {
        let range = fbs::TimeRange::new(0, 1_000);
        let q = fbs::QuoteRequest::create(
            b,
            &fbs::QuoteRequestArgs {
                band: continuum_core::PriorityBand::P2Nearline,
                compute_domain: continuum_core::ComputeDomain::CPU,
                est_duration_ms: 10,
                capability_id: cap,
                range: Some(&range),
                ..Default::default()
            },
        );
        (ControlPayload::QuoteRequest, q.as_union_value())
    });
    c.send(&quote).await;
    let (_, kind, bytes) = c.recv().await;
    assert_eq!(kind, ControlPayload::LeaseGrant);
    let g = read_envelope(&bytes).unwrap().payload_as_lease_grant().unwrap();
    assert!(g.granted());
    let lease = g.lease().unwrap().lease_id();
    let header = header(7, cam);
    let commit = envelope(5, |b| {
        let h = b.create_vector(&header);
        let item = fbs::Item::create(b, &fbs::ItemArgs { locator: None, header_bytes: Some(h), commit_seq: 0 });
        let outputs = b.create_vector(&[item]);
        let c = fbs::CommitLease::create(
            b,
            &fbs::CommitLeaseArgs { lease_id: lease, outputs: Some(outputs), coverage: None, cost: None },
        );
        (ControlPayload::CommitLease, c.as_union_value())
    });
    c.send(&commit).await;
    let seqs = acked(&mut c, 5, lease).await;
    // Indexed: the same payload again through the ingest path is the repeat of a commit, with its sequence.
    c.send(&ingest(6, cam, &header)).await;
    assert_eq!(ingested(&mut c).await.3, seqs[0]);
    broker.stop().await;
}

// ---- The Tier 0 arena ----

/// Says Hello and maps the arena Welcome passes: (the arena, its name, its generation).
async fn welcome_arena(c: &mut Client) -> (continuum_shm::Arena, String, u32) {
    use std::os::fd::AsFd;
    c.send(&hello(1, SCHEMA_MAJOR)).await;
    let fds =
        tokio::time::timeout(Duration::from_secs(5), continuum_wire::read_frame_with_fds(&mut c.stream, &mut c.buf, 4))
            .await
            .expect("Welcome within 5 s")
            .expect("a frame");
    let env = read_envelope(&c.buf).unwrap();
    assert_eq!(env.payload_type(), ControlPayload::Welcome);
    let arenas = env.payload_as_welcome().unwrap().arenas().unwrap();
    assert_eq!((arenas.len(), fds.len()), (1, 1), "one arena, one fd");
    let info = arenas.get(0);
    let name = info.name().unwrap().to_owned();
    let classes: Vec<_> = info.classes().unwrap().iter().map(|c| (c.slab_bytes(), c.count())).collect();
    assert_eq!(classes, [(16_384, 4), (65_536, 2)]);
    let arena = continuum_shm::Arena::from_fd(fds[0].as_fd(), &name).unwrap();
    assert_eq!(arena.len() as u64, info.size_bytes());
    (arena, name, info.generation())
}

fn shm_ingest(request_id: u64, stream: u32, n: u8, locator: &continuum_core::StorageLocator) -> Vec<u8> {
    let header = header(n, stream);
    envelope(request_id, |b| {
        let h = b.create_vector(&header);
        let loc = flatbuffers_locator(b, locator);
        let notice = IngestNotice::create(
            b,
            &IngestNoticeArgs { stream_id: stream, header_bytes: Some(h), locator: Some(loc), ..Default::default() },
        );
        (ControlPayload::IngestNotice, notice.as_union_value())
    })
}

/// Claims a slot in class 0, writes `len` bytes, and publishes it: its Tier 0 locator.
fn publish(arena: &continuum_shm::Arena, name: &str, generation: u32, len: usize) -> continuum_core::StorageLocator {
    let mut slot = arena.claim(0, 1).unwrap();
    slot.as_mut_slice()[..len].fill(7);
    let offset = slot.offset() as u64;
    let h = slot.publish(len).unwrap();
    continuum_core::StorageLocator::Shm {
        arena: name.to_owned(),
        arena_generation: generation,
        slab_class: h.class,
        slot: h.index,
        slot_generation: h.generation,
        offset,
        length: len as u64,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn welcome_passes_the_arena_and_a_published_slot_ingests_as_a_tier_0_copy() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;
    let (arena, name, generation) = welcome_arena(&mut c).await;
    c.send(&register_stream(2, "cam0", 30.0)).await;
    let (_, cam) = stream_registered(&mut c).await;
    let loc = publish(&arena, &name, generation, 100);
    // Request id 0: indexed silently (a numbered notice's Ingested reply could arrive before the heartbeat's).
    c.send(&shm_ingest(0, cam, 1, &loc)).await;
    committed(&broker, &mut c).await;
    let covered = broker.core.covered(continuum_core::StreamId(cam), Interval::new(0, 1000).unwrap()).await.unwrap();
    assert_eq!(covered.len(), 1);
    assert_eq!(covered[0].tier, continuum_core::StorageTier::Tier0Hot);
    // The check's own pin was released.
    let continuum_core::StorageLocator::Shm { slab_class, slot, slot_generation, .. } = loc else { unreachable!() };
    let h = continuum_core::SlotHandle { class: slab_class, index: slot, generation: slot_generation };
    assert_eq!(arena.pin_count(h).unwrap(), 0);
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_tier_0_locator_that_doesnt_name_a_published_slot_is_rejected() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;
    let (arena, name, generation) = welcome_arena(&mut c).await;
    c.send(&register_stream(2, "cam0", 30.0)).await;
    let (_, cam) = stream_registered(&mut c).await;
    let good = publish(&arena, &name, generation, 100);
    let with = |f: &dyn Fn(&mut continuum_core::StorageLocator)| {
        let mut l = good.clone();
        f(&mut l);
        l
    };
    use continuum_core::StorageLocator::Shm;
    let held = arena.claim(0, 1).unwrap(); // claimed, never published
    let unpublished = Shm {
        arena: name.clone(),
        arena_generation: generation,
        slab_class: held.handle().class,
        slot: held.handle().index,
        slot_generation: held.handle().generation,
        offset: held.offset() as u64,
        length: 0,
    };
    let cases: Vec<(continuum_core::StorageLocator, &str)> = vec![
        (
            with(&|l| {
                if let Shm { arena_generation, .. } = l {
                    *arena_generation += 1
                }
            }),
            "names arena",
        ),
        (
            with(&|l| {
                if let Shm { arena, .. } = l {
                    *arena = "/cnt.other.1".into()
                }
            }),
            "names arena",
        ),
        (
            with(&|l| {
                if let Shm { slot, .. } = l {
                    *slot = 99
                }
            }),
            "doesn't exist",
        ),
        (
            with(&|l| {
                if let Shm { slab_class, .. } = l {
                    *slab_class = 9
                }
            }),
            "doesn't exist",
        ),
        (
            with(&|l| {
                if let Shm { offset, .. } = l {
                    *offset += 16_384
                }
            }),
            "offset",
        ),
        (
            with(&|l| {
                if let Shm { slot_generation, .. } = l {
                    *slot_generation += 1
                }
            }),
            "isn't published",
        ),
        (
            with(&|l| {
                if let Shm { length, .. } = l {
                    *length = 99
                }
            }),
            "length",
        ),
        (unpublished, "isn't published"),
    ];
    for (i, (loc, why)) in cases.iter().enumerate() {
        let id = 10 + i as u64;
        c.send(&shm_ingest(id, cam, 1, loc)).await;
        let (got, reason) = rejected(&mut c).await;
        assert_eq!(got, id);
        assert!(reason.contains(why), "case {i}: {reason}");
    }
    // The good one still goes in after all that.
    c.send(&shm_ingest(0, cam, 1, &good)).await;
    committed(&broker, &mut c).await;
    let covered = broker.core.covered(continuum_core::StreamId(cam), Interval::new(0, 1000).unwrap()).await.unwrap();
    assert_eq!(covered.len(), 1);
    drop(held);
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_arena_is_unlinked_on_stop_and_a_crashed_runs_arena_on_the_next_start() {
    use continuum_broker::tier0::{arena_name, generation_path};
    let broker = Broker::start(Duration::from_secs(30)).await;
    let socket = broker.socket.clone();
    let mut c = Client::connect(&socket).await;
    let (_arena, name, first) = welcome_arena(&mut c).await;
    assert_eq!(name, arena_name(&socket, first));
    let clock =
        u32::try_from(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()).unwrap();
    assert!(first + 1 >= clock, "seeded from the clock, so it rises even without the generation file");
    drop(c);
    let dir = broker.stop().await;
    assert!(continuum_shm::Arena::open(&name).is_err(), "unlinked on stop");

    // A run that crashed (at a generation ahead of the clock, as after a quick restart) left its arena behind.
    let crashed = first + 1_000;
    std::fs::write(generation_path(&socket), format!("{crashed}\n")).unwrap();
    let left = arena_name(&socket, crashed);
    let config = continuum_shm::ArenaConfig {
        classes: vec![continuum_shm::SlabClass { slot_size: 16_384, slot_count: 1 }],
        ..Default::default()
    };
    drop(continuum_shm::Arena::create(&left, &config).unwrap());
    let server = Server::bind(test_config(socket.clone(), Duration::from_secs(30))).await.unwrap();
    assert!(continuum_shm::Arena::open(&left).is_err(), "the crashed run's arena is gone");
    let next = arena_name(&socket, crashed + 1);
    assert!(continuum_shm::Arena::open(&next).is_ok(), "the new one is the next generation");
    drop(server);
    assert!(continuum_shm::Arena::open(&next).is_err());

    // With the generation file lost, the clock still keeps it from going back to 1 (only a restart within the same
    // second as a run it can't remember could repeat that run's generation).
    std::fs::remove_file(generation_path(&socket)).unwrap();
    let server = Server::bind(test_config(socket.clone(), Duration::from_secs(30))).await.unwrap();
    let file = std::fs::read_to_string(generation_path(&socket)).unwrap();
    let g: u32 = file.lines().next().unwrap().parse().unwrap();
    assert_eq!(file.lines().nth(1), Some(arena_name(&socket, g).as_str()), "the file names the arena too");
    assert!(g >= clock, "{g} is at least the clock's {clock}");
    drop(server);
    drop(dir);
}

/// Waits for the `LeaseReleased` that answers a commit or an abort.
/// The commit's `LeaseReleased`: its outputs' commit sequences, each positive (none hidden in these tests).
async fn acked(c: &mut Client, request_id: u64, lease: u64) -> Vec<u64> {
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (request_id, ControlPayload::LeaseReleased));
    let r = read_envelope(&bytes).unwrap().payload_as_lease_released().unwrap();
    assert_eq!(r.lease_id(), lease);
    let seqs: Vec<u64> = r.output_seqs().map(|v| v.iter().collect()).unwrap_or_default();
    assert!(seqs.iter().all(|&s| s > 0), "every output has a commit sequence: {seqs:?}");
    seqs
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_commit_is_answered_once_durable_a_failed_one_by_its_request_id_and_an_abort_too() {
    use continuum_core::PriorityBand;
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut w, cap, pose, _) = worker(&broker).await;
    let output = |n: u8, start: u64| {
        let mut id = [0u8; 16];
        id[6] = 0x70;
        id[15] = n;
        let spec = continuum_core::HeaderSpec::raw(
            continuum_core::Uuid(id),
            continuum_core::StreamId(pose),
            start,
            start + 10,
        );
        continuum_core::HeaderBuf::build(&spec).unwrap().into_bytes()
    };
    let commit = |request_id: u64, lease: u64, header: Vec<u8>| {
        envelope(request_id, |b| {
            let h = b.create_vector(&header);
            let loc = flatbuffers_locator(b, &chunk_at(continuum_core::StorageTier::Tier1Warm));
            let item =
                fbs::Item::create(b, &fbs::ItemArgs { locator: Some(loc), header_bytes: Some(h), commit_seq: 0 });
            let outputs = b.create_vector(&[item]);
            let c = fbs::CommitLease::create(
                b,
                &fbs::CommitLeaseArgs { lease_id: lease, outputs: Some(outputs), coverage: None, cost: None },
            );
            (ControlPayload::CommitLease, c.as_union_value())
        })
    };
    // The ack means durable: the output is in the index the moment it arrives, with no sync in between.
    let lease = grant(&mut w, 5, cap, PriorityBand::P2Nearline).await;
    w.send(&commit(6, lease, output(1, 100))).await;
    acked(&mut w, 6, lease).await;
    let got = broker.core.covered(continuum_core::StreamId(pose), Interval::new(0, 1_000).unwrap()).await.unwrap();
    assert_eq!(got.len(), 1, "indexed by the time the ack arrives");
    // A different payload at the same time conflicts in the writer, and the late error still carries the request id.
    let lease = grant(&mut w, 7, cap, PriorityBand::P2Nearline).await;
    w.send(&commit(8, lease, output(2, 100))).await;
    let (id, code, _) = w.recv_error().await;
    assert_eq!((id, code), (8, ErrorCode::Conflict));
    // An abort is answered too.
    let lease = grant(&mut w, 9, cap, PriorityBand::P2Nearline).await;
    let abort = envelope(10, |b| {
        let a = fbs::AbortLease::create(b, &fbs::AbortLeaseArgs { lease_id: lease, reason: None });
        (ControlPayload::AbortLease, a.as_union_value())
    });
    w.send(&abort).await;
    acked(&mut w, 10, lease).await;
    broker.stop().await;
}

/// A `CommitLease` of one Tier 1 output, with coverage of `source` if given.
fn commit_with(request_id: u64, lease: u64, header: &[u8], coverage_source: Option<u32>) -> Vec<u8> {
    envelope(request_id, |b| {
        let h = b.create_vector(header);
        let loc = flatbuffers_locator(b, &chunk_at(continuum_core::StorageTier::Tier1Warm));
        let item = fbs::Item::create(b, &fbs::ItemArgs { locator: Some(loc), header_bytes: Some(h), commit_seq: 0 });
        let outputs = b.create_vector(&[item]);
        let coverage = coverage_source.map(|source| {
            let stage = b.create_string("pose");
            let range = fbs::TimeRange::new(0, 1_000);
            fbs::CoverageDone::create(
                b,
                &fbs::CoverageDoneArgs { source_stream_id: source, stage: Some(stage), range: Some(&range) },
            )
        });
        let c = fbs::CommitLease::create(
            b,
            &fbs::CommitLeaseArgs { lease_id: lease, outputs: Some(outputs), coverage, cost: None },
        );
        (ControlPayload::CommitLease, c.as_union_value())
    })
}

fn output_on(stream: u32, n: u8, start: u64) -> Vec<u8> {
    let mut id = [0u8; 16];
    id[6] = 0x70;
    id[15] = n;
    let spec =
        continuum_core::HeaderSpec::raw(continuum_core::Uuid(id), continuum_core::StreamId(stream), start, start + 10);
    continuum_core::HeaderBuf::build(&spec).unwrap().into_bytes()
}

/// The next `n` replies, by request id.
async fn replies(c: &mut Client, n: usize) -> std::collections::HashMap<u64, (ControlPayload, Vec<u8>)> {
    let mut out = std::collections::HashMap::new();
    for _ in 0..n {
        let (id, kind, bytes) = c.recv().await;
        out.insert(id, (kind, bytes));
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_second_commit_or_an_abort_racing_a_queued_commit_is_refused_not_also_acked() {
    use continuum_core::PriorityBand;
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut w, cap, pose, _) = worker(&broker).await;
    // Two commits sent together: the first is acked, the second refused before it stores anything.
    let lease = grant(&mut w, 5, cap, PriorityBand::P2Nearline).await;
    let mut both = commit_with(6, lease, &output_on(pose, 1, 100), None);
    let second = commit_with(7, lease, &output_on(pose, 2, 200), None);
    let mut frames = Vec::new();
    for f in [&mut both, &mut second.clone()] {
        frames.extend_from_slice(&u32::try_from(f.len()).unwrap().to_le_bytes());
        frames.extend_from_slice(f);
    }
    tokio::io::AsyncWriteExt::write_all(&mut w.stream, &frames).await.unwrap();
    let got = replies(&mut w, 2).await;
    assert_eq!(got[&6].0, ControlPayload::LeaseReleased);
    assert_eq!(got[&7].0, ControlPayload::Error);
    let got = broker.core.covered(continuum_core::StreamId(pose), Interval::new(0, 1_000).unwrap()).await.unwrap();
    assert_eq!(got.len(), 1, "only the first commit's output: {got:?}");
    // A commit then an abort: the abort is refused.
    let lease = grant(&mut w, 8, cap, PriorityBand::P2Nearline).await;
    w.send(&commit_with(9, lease, &output_on(pose, 3, 300), None)).await;
    let abort = envelope(10, |b| {
        let a = fbs::AbortLease::create(b, &fbs::AbortLeaseArgs { lease_id: lease, reason: None });
        (ControlPayload::AbortLease, a.as_union_value())
    });
    w.send(&abort).await;
    let got = replies(&mut w, 2).await;
    assert_eq!(got[&9].0, ControlPayload::LeaseReleased);
    assert_eq!(got[&10].0, ControlPayload::Error);
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_renewal_extends_by_the_requested_time_capped_at_the_maximum_lease() {
    use continuum_core::PriorityBand;
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut w, cap, _, _) = worker(&broker).await;
    let lease = grant(&mut w, 5, cap, PriorityBand::P2Nearline).await;
    let wall = || {
        u64::try_from(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()).unwrap()
    };
    let mut renew = async |request_id: u64, extend_ms: u32| {
        let r = envelope(request_id, |b| {
            let r = fbs::RenewLease::create(b, &fbs::RenewLeaseArgs { lease_id: lease, extend_ms });
            (ControlPayload::RenewLease, r.as_union_value())
        });
        w.send(&r).await;
        let (id, kind, bytes) = w.recv().await;
        assert_eq!((id, kind), (request_id, ControlPayload::LeaseGrant));
        read_envelope(&bytes).unwrap().payload_as_lease_grant().unwrap().lease().unwrap().expires_at_ns()
    };
    let s = 1_000_000_000u64;
    let near = |got: u64, want: u64| got.abs_diff(want) < 5 * s;
    let two_minutes = renew(6, 120_000).await;
    assert!(near(two_minutes, wall() + 120 * s), "asked for 2 minutes");
    // 0 asks for the default 10 s, which never pulls an expiry earlier.
    assert_eq!(renew(7, 0).await, two_minutes);
    // An hour is capped at the 5 minute maximum.
    assert!(near(renew(8, 3_600_000).await, wall() + 300 * s), "capped at 5 minutes");
    broker.stop().await;
}

fn query_for(request_id: u64, stream: &str, start: u64, end: u64, mode: fbs::QueryMode) -> Vec<u8> {
    query_capped(request_id, stream, start, end, mode, 0)
}

fn query_capped(request_id: u64, stream: &str, start: u64, end: u64, mode: fbs::QueryMode, max_items: u32) -> Vec<u8> {
    envelope(request_id, |b| {
        let name = b.create_string(stream);
        let range = fbs::TimeRange::new(start, end);
        let q = Query::create(
            b,
            &QueryArgs {
                stream: Some(name),
                range: Some(&range),
                band: continuum_core::PriorityBand::P2Nearline,
                mode,
                max_items,
                ..Default::default()
            },
        );
        (ControlPayload::Query, q.as_union_value())
    })
}

fn commit_promise(request_id: u64, promise: [u8; 16], option: u32) -> Vec<u8> {
    envelope(request_id, |b| {
        let id = fbs::Uuid::new(&promise);
        let c = fbs::CommitPromise::create(
            b,
            &fbs::CommitPromiseArgs { promise_id: Some(&id), option, band: continuum_core::PriorityBand::P2Nearline },
        );
        (ControlPayload::CommitPromise, c.as_union_value())
    })
}

fn cancel_promise(request_id: u64, promise: [u8; 16]) -> Vec<u8> {
    envelope(request_id, |b| {
        let id = fbs::Uuid::new(&promise);
        let c = fbs::CancelPromise::create(b, &fbs::CancelPromiseArgs { promise_id: Some(&id) });
        (ControlPayload::CancelPromise, c.as_union_value())
    })
}

fn job_rejected(request_id: u64, job: [u8; 16], reason: &str) -> Vec<u8> {
    envelope(request_id, |b| {
        let id = fbs::Uuid::new(&job);
        let reason = b.create_string(reason);
        let j = fbs::JobRejected::create(b, &fbs::JobRejectedArgs { job_id: Some(&id), reason: Some(reason) });
        (ControlPayload::JobRejected, j.as_union_value())
    })
}

/// A consumer: a session of its own that only asks.
async fn consumer(broker: &Broker) -> Client {
    let mut c = Client::connect(&broker.socket).await;
    c.send(&hello(1, SCHEMA_MAJOR)).await;
    assert_eq!(c.recv().await.1, ControlPayload::Welcome);
    c
}

/// A `QueryResult`: (items, complete, missing).
fn query_result(bytes: &[u8]) -> (usize, bool, Vec<(u64, u64)>) {
    let r = read_envelope(bytes).unwrap().payload_as_query_result().unwrap();
    let missing = r.missing().map(|m| m.iter().map(|t| (t.start_ns(), t.end_ns())).collect()).unwrap_or_default();
    (r.items().map_or(0, |i| i.len()), r.complete(), missing)
}

/// Three cam0 frames over [100, 400), with copies in `tier` (Tier 0 slots for a compute step's inputs, since a
/// Tier 1 copy would need a hydrate capability first).
async fn three_frames(broker: &Broker, p: &mut Client, cam: u32, tier: continuum_core::StorageTier) {
    for (n, start) in [(1u8, 100u64), (2, 200), (3, 300)] {
        let locator = if tier == continuum_core::StorageTier::Tier0Hot {
            // A real slot in the broker's arena: the broker checks every Tier 0 locator.
            let (arena, name, generation) = broker.arena();
            publish(&arena, &name, generation, 1)
        } else {
            chunk_at(tier)
        };
        p.send(&ingest_located(0, cam, n, start, start + 100, &locator)).await;
    }
    committed(broker, p).await;
}

fn ingest_located(
    request_id: u64,
    stream: u32,
    n: u8,
    start: u64,
    end: u64,
    locator: &continuum_core::StorageLocator,
) -> Vec<u8> {
    let spec =
        continuum_core::HeaderSpec::raw(continuum_core::Uuid(id_of(n)), continuum_core::StreamId(stream), start, end);
    let header = continuum_core::HeaderBuf::build(&spec).unwrap().into_bytes();
    envelope(request_id, |b| {
        let h = b.create_vector(&header);
        let loc = flatbuffers_locator(b, locator);
        let n = IngestNotice::create(
            b,
            &IngestNoticeArgs { stream_id: stream, header_bytes: Some(h), locator: Some(loc), ..Default::default() },
        );
        (ControlPayload::IngestNotice, n.as_union_value())
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_hit_query_answers_with_what_exists_and_what_is_missing() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut p, cam) = producer(&broker).await;
    three_frames(&broker, &mut p, cam, continuum_core::StorageTier::Tier1Warm).await;
    let mut c = consumer(&broker).await;

    c.send(&query_for(7, "cam0", 100, 400, fbs::QueryMode::HitOnly)).await;
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (7, ControlPayload::QueryResult));
    assert_eq!(query_result(&bytes), (3, true, vec![]));
    let r = read_envelope(&bytes).unwrap().payload_as_query_result().unwrap();
    assert!(!r.quoted(), "a hit-only query is never quoted");
    let first = r.items().unwrap().get(0);
    assert_eq!(first.locator().unwrap().tier(), continuum_core::StorageTier::Tier1Warm);
    let header = continuum_core::HeaderBuf::new(first.header_bytes().unwrap().bytes().to_vec()).unwrap();
    assert_eq!(header.stream(), continuum_core::StreamId(cam));

    c.send(&query_for(8, "cam0", 100, 600, fbs::QueryMode::HitOnly)).await;
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (8, ControlPayload::QueryResult));
    assert_eq!(query_result(&bytes), (3, false, vec![(400, 600)]));

    c.send(&query_for(9, "nope", 0, 10, fbs::QueryMode::HitOnly)).await;
    let (id, code, message) = c.recv_error().await;
    assert_eq!((id, code), (9, ErrorCode::NotFound));
    assert!(message.contains("nope"), "{message}");
    c.send(&query_for(10, "cam0", 10, 10, fbs::QueryMode::HitOnly)).await;
    assert_eq!(c.recv_error().await.1, ErrorCode::BadRequest);
    broker.stop().await;
}

/// Registers the worker and its streams, ingests three cam0 frames, and has a consumer commit a quote for pose
/// over them: returns the worker, the consumer, the promise id, and the DispatchJob the worker received.
async fn dispatched(broker: &Broker) -> (Client, Client, [u8; 16], Vec<u8>, u32) {
    let (mut w, cap, pose, cam) = worker(broker).await;
    three_frames(broker, &mut w, cam, continuum_core::StorageTier::Tier0Hot).await;
    let mut c = consumer(broker).await;

    c.send(&query_for(7, "pose", 100, 400, fbs::QueryMode::QuoteIfMissing)).await;
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (7, ControlPayload::QueryResult));
    assert_eq!(query_result(&bytes), (0, false, vec![(100, 400)]));
    assert!(read_envelope(&bytes).unwrap().payload_as_query_result().unwrap().quoted(), "it says a quote follows");
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (7, ControlPayload::PromiseQuote), "the quote follows with the query's id");
    let q = read_envelope(&bytes).unwrap().payload_as_promise_quote().unwrap();
    assert_eq!(q.status(), fbs::PromiseStatus::RequiresCompute);
    let option = q.options().unwrap().get(0);
    let steps: Vec<(String, String)> = option
        .steps()
        .unwrap()
        .iter()
        .map(|s| (s.worker_id().unwrap_or("").to_owned(), s.produces().unwrap_or("").to_owned()))
        .collect();
    assert_eq!(steps, vec![("pose-worker".to_owned(), "pose".to_owned())]);
    assert!(q.expires_at_ns() > 0);
    let promise: [u8; 16] = q.promise_id().unwrap().bytes().into();

    c.send(&commit_promise(8, promise, 0)).await;
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (8, ControlPayload::PromiseAccepted));
    let a = read_envelope(&bytes).unwrap().payload_as_promise_accepted().unwrap();
    assert_eq!(a.life(), fbs::PromiseLife::Dispatching);

    let (id, kind, bytes) = w.recv().await;
    assert_eq!((id, kind), (0, ControlPayload::DispatchJob), "pushed to the worker");
    let d = read_envelope(&bytes).unwrap().payload_as_dispatch_job().unwrap();
    assert_eq!(d.capability_id(), cap);
    assert_eq!(d.capability().unwrap(), "pose");
    assert_eq!(d.inputs().map_or(0, |i| i.len()), 3, "the cam frames, resolved and handed over");
    assert!(d.promise_ids().unwrap().iter().any(|p| p.bytes().into_iter().collect::<Vec<u8>>() == promise));
    let range = d.output_range().unwrap();
    assert_eq!((range.start_ns(), range.end_ns()), (100, 400));
    assert!(d.output().unwrap().lease_id() > 0);
    (w, c, promise, bytes, pose)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_quoted_query_commits_dispatches_and_fulfills() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut w, mut c, promise, dispatch, pose) = dispatched(&broker).await;
    let d = read_envelope(&dispatch).unwrap().payload_as_dispatch_job().unwrap();
    let lease = d.output().unwrap().lease_id();

    // The worker commits one pose payload over the whole range.
    let stream = continuum_core::StreamId(pose);
    let spec = continuum_core::HeaderSpec::raw(continuum_core::Uuid(id_of(77)), stream, 100, 400);
    let header = continuum_core::HeaderBuf::build(&spec).unwrap().into_bytes();
    let commit = envelope(9, |b| {
        let h = b.create_vector(&header);
        let loc = flatbuffers_locator(b, &chunk_at(continuum_core::StorageTier::Tier1Warm));
        let item = fbs::Item::create(b, &fbs::ItemArgs { locator: Some(loc), header_bytes: Some(h), commit_seq: 0 });
        let outputs = b.create_vector(&[item]);
        let c = fbs::CommitLease::create(
            b,
            &fbs::CommitLeaseArgs { lease_id: lease, outputs: Some(outputs), coverage: None, cost: None },
        );
        (ControlPayload::CommitLease, c.as_union_value())
    });
    w.send(&commit).await;
    acked(&mut w, 9, lease).await;

    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (0, ControlPayload::FulfillPromise), "pushed to the consumer");
    let f = read_envelope(&bytes).unwrap().payload_as_fulfill_promise().unwrap();
    assert_eq!(f.promise_id().unwrap().bytes().into_iter().collect::<Vec<u8>>(), promise);
    assert!(f.final_part());
    let items = f.items().unwrap();
    assert_eq!(items.len(), 1);
    let got = continuum_core::HeaderBuf::new(items.get(0).header_bytes().unwrap().bytes().to_vec()).unwrap();
    assert_eq!(got.id(), continuum_core::Uuid(id_of(77)));
    broker.stop().await;
}

/// Two consumers asking for the same missing range: the second quote says it shares the job in flight,
/// committing it sends the worker nothing new, and the one job's commit fulfills both promises.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_second_consumer_shares_the_job_in_flight() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut w, mut c, first, dispatch, pose) = dispatched(&broker).await;
    let lease = read_envelope(&dispatch).unwrap().payload_as_dispatch_job().unwrap().output().unwrap().lease_id();

    let mut c2 = consumer(&broker).await;
    c2.send(&query_for(20, "pose", 100, 400, fbs::QueryMode::QuoteIfMissing)).await;
    assert_eq!(c2.recv().await.1, ControlPayload::QueryResult);
    let (id, kind, bytes) = c2.recv().await;
    assert_eq!((id, kind), (20, ControlPayload::PromiseQuote));
    let q = read_envelope(&bytes).unwrap().payload_as_promise_quote().unwrap();
    assert!(q.shares_in_flight(), "the quote says the work is already running");
    let second: [u8; 16] = q.promise_id().unwrap().bytes().into();
    assert_ne!(second, first);
    c2.send(&commit_promise(21, second, 0)).await;
    let (id, kind, _) = c2.recv().await;
    assert_eq!((id, kind), (21, ControlPayload::PromiseAccepted));

    // The worker commits the one job's output.
    let stream = continuum_core::StreamId(pose);
    let spec = continuum_core::HeaderSpec::raw(continuum_core::Uuid(id_of(78)), stream, 100, 400);
    let header = continuum_core::HeaderBuf::build(&spec).unwrap().into_bytes();
    let commit = envelope(22, |b| {
        let h = b.create_vector(&header);
        let loc = flatbuffers_locator(b, &chunk_at(continuum_core::StorageTier::Tier1Warm));
        let item = fbs::Item::create(b, &fbs::ItemArgs { locator: Some(loc), header_bytes: Some(h), commit_seq: 0 });
        let outputs = b.create_vector(&[item]);
        let c = fbs::CommitLease::create(
            b,
            &fbs::CommitLeaseArgs { lease_id: lease, outputs: Some(outputs), coverage: None, cost: None },
        );
        (ControlPayload::CommitLease, c.as_union_value())
    });
    w.send(&commit).await;
    acked(&mut w, 22, lease).await; // the next thing the worker heard: no second DispatchJob came first
    assert!(
        tokio::time::timeout(Duration::from_millis(500), w.recv()).await.is_err(),
        "nor after: the worker hears nothing more"
    );

    for (consumer, promise) in [(&mut c, first), (&mut c2, second)] {
        let (id, kind, bytes) = consumer.recv().await;
        assert_eq!((id, kind), (0, ControlPayload::FulfillPromise));
        let f = read_envelope(&bytes).unwrap().payload_as_fulfill_promise().unwrap();
        assert_eq!(f.promise_id().unwrap().bytes().into_iter().collect::<Vec<u8>>(), promise);
        let items = f.items().unwrap();
        let got = continuum_core::HeaderBuf::new(items.get(0).header_bytes().unwrap().bytes().to_vec()).unwrap();
        assert_eq!(got.id(), continuum_core::Uuid(id_of(78)), "both get the one job's output");
    }
    broker.stop().await;
}

fn id_of(n: u8) -> [u8; 16] {
    let mut id = [0u8; 16];
    id[6] = 0x70;
    id[15] = n;
    id
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelling_a_dispatched_promise_revokes_its_job() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut w, mut c, promise, _, _) = dispatched(&broker).await;
    c.send(&cancel_promise(10, promise)).await;
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (10, ControlPayload::PromiseAccepted));
    assert_eq!(
        read_envelope(&bytes).unwrap().payload_as_promise_accepted().unwrap().life(),
        fbs::PromiseLife::Cancelled
    );
    let (id, kind, bytes) = w.recv().await;
    assert_eq!((id, kind), (0, ControlPayload::LeasePreempted));
    let p = read_envelope(&bytes).unwrap().payload_as_lease_preempted().unwrap();
    assert!(p.reason().unwrap_or("").starts_with("revoked:"), "{:?}", p.reason());
    // A second cancel: the promise is gone.
    c.send(&cancel_promise(11, promise)).await;
    assert_eq!(c.recv_error().await.1, ErrorCode::NotFound);
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_rejected_job_fails_the_promise_and_only_its_worker_may_reject_it() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut w, mut c, promise, dispatch, _) = dispatched(&broker).await;
    let d = read_envelope(&dispatch).unwrap().payload_as_dispatch_job().unwrap();
    let job: [u8; 16] = d.job_id().unwrap().bytes().into();
    // The consumer can't reject a job (it registered no worker); nothing happens.
    c.send(&job_rejected(12, job, "not mine")).await;
    c.send(&heartbeat(13)).await;
    let (id, kind, _) = c.recv().await;
    assert_eq!((id, kind), (13, ControlPayload::Heartbeat), "nothing else came back");
    // The worker can. The engine replans a rejected job onto the only producer twice (Settings::max_replans) before
    // it gives up, so the worker sees two more dispatches to reject.
    w.send(&job_rejected(14, job, "no model loaded")).await;
    for i in 0..2 {
        let (id, kind, bytes) = w.recv().await;
        assert_eq!((id, kind), (0, ControlPayload::DispatchJob), "replan {i}");
        let again: [u8; 16] =
            read_envelope(&bytes).unwrap().payload_as_dispatch_job().unwrap().job_id().unwrap().bytes().into();
        assert_ne!(again, job);
        w.send(&job_rejected(15 + i, again, "no model loaded")).await;
    }
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (0, ControlPayload::PromiseFailed));
    let f = read_envelope(&bytes).unwrap().payload_as_promise_failed().unwrap();
    assert_eq!(f.promise_id().unwrap().bytes().into_iter().collect::<Vec<u8>>(), promise);
    assert_eq!(f.delivered(), 0);
    assert!(f.reason().unwrap_or("").contains("no model"), "{:?}", f.reason());
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_future_range_on_a_live_stream_arms_and_fulfills_when_captured() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut p, cam) = producer(&broker).await;
    let mut c = consumer(&broker).await;
    let now =
        u64::try_from(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()).unwrap();
    let (start, end) = (now + 2_000_000_000, now + 2_200_000_000);

    c.send(&query_for(7, "cam0", start, end, fbs::QueryMode::QuoteIfMissing)).await;
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (7, ControlPayload::QueryResult));
    assert_eq!(query_result(&bytes), (0, false, vec![(start, end)]));
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (7, ControlPayload::PromiseQuote));
    let q = read_envelope(&bytes).unwrap().payload_as_promise_quote().unwrap();
    assert_eq!(q.status(), fbs::PromiseStatus::AwaitingCapture);
    let promise: [u8; 16] = q.promise_id().unwrap().bytes().into();
    c.send(&commit_promise(8, promise, 0)).await;
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (8, ControlPayload::PromiseAccepted));
    assert_eq!(read_envelope(&bytes).unwrap().payload_as_promise_accepted().unwrap().life(), fbs::PromiseLife::Armed);

    // The capture lands, in two frames covering the range.
    p.send(&ingest_copy(0, cam, 1, start, start + 100_000_000, &[])).await;
    p.send(&ingest_copy(0, cam, 2, start + 100_000_000, end, &[])).await;
    committed(&broker, &mut p).await;
    // Each frame that lands is a part; the one that completes the range is final.
    let mut finals = Vec::new();
    for _ in 0..2 {
        let (id, kind, bytes) = c.recv().await;
        assert_eq!((id, kind), (0, ControlPayload::FulfillPromise));
        let f = read_envelope(&bytes).unwrap().payload_as_fulfill_promise().unwrap();
        assert_eq!(f.promise_id().unwrap().bytes().into_iter().collect::<Vec<u8>>(), promise);
        assert_eq!(f.items().unwrap().len(), 1);
        finals.push(f.final_part());
    }
    assert_eq!(finals, vec![false, true]);
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_result_cut_at_max_items_is_not_complete() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut p, cam) = producer(&broker).await;
    three_frames(&broker, &mut p, cam, continuum_core::StorageTier::Tier1Warm).await;
    let mut c = consumer(&broker).await;
    c.send(&query_capped(7, "cam0", 100, 400, fbs::QueryMode::HitOnly, 2)).await;
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (7, ControlPayload::QueryResult));
    assert_eq!(query_result(&bytes), (2, false, vec![]), "the earliest two, and not complete");
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn only_the_session_that_was_quoted_may_commit_and_an_expired_quote_is_forgotten() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut w, _cap, _pose, cam) = worker(&broker).await;
    three_frames(&broker, &mut w, cam, continuum_core::StorageTier::Tier0Hot).await;
    let mut c = consumer(&broker).await;
    c.send(&query_for(7, "pose", 100, 400, fbs::QueryMode::QuoteIfMissing)).await;
    assert_eq!(c.recv().await.1, ControlPayload::QueryResult);
    let (_, kind, bytes) = c.recv().await;
    assert_eq!(kind, ControlPayload::PromiseQuote);
    let promise: [u8; 16] =
        read_envelope(&bytes).unwrap().payload_as_promise_quote().unwrap().promise_id().unwrap().bytes().into();

    // Another session: not its promise.
    let mut other = consumer(&broker).await;
    other.send(&commit_promise(8, promise, 0)).await;
    assert_eq!(other.recv_error().await.1, ErrorCode::NotAllowed);
    other.send(&cancel_promise(9, promise)).await;
    assert_eq!(other.recv_error().await.1, ErrorCode::NotAllowed);

    // The quote expires (2 s) and the broker's next tick forgets it, so a late commit finds no promise at all (the
    // owner map doesn't keep expired quotes).
    tokio::time::sleep(Duration::from_millis(2_300)).await;
    c.send(&commit_promise(10, promise, 0)).await;
    let (id, code, _) = c.recv_error().await;
    assert_eq!((id, code), (10, ErrorCode::NotFound), "an expired quote is nobody's");
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_query_within_budget_commits_at_once_and_dispatches() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut w, cap, _pose, cam) = worker(&broker).await;
    three_frames(&broker, &mut w, cam, continuum_core::StorageTier::Tier0Hot).await;
    let mut c = consumer(&broker).await;
    c.send(&query_for(7, "pose", 100, 400, fbs::QueryMode::CommitWithinBudget)).await;
    let (id, kind, _) = c.recv().await;
    assert_eq!((id, kind), (7, ControlPayload::QueryResult));
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (7, ControlPayload::PromiseQuote));
    let promise: [u8; 16] =
        read_envelope(&bytes).unwrap().payload_as_promise_quote().unwrap().promise_id().unwrap().bytes().into();
    // No CommitPromise needed: the job is already with the worker.
    let (id, kind, bytes) = w.recv().await;
    assert_eq!((id, kind), (0, ControlPayload::DispatchJob));
    let d = read_envelope(&bytes).unwrap().payload_as_dispatch_job().unwrap();
    assert_eq!(d.capability_id(), cap);
    assert!(d.promise_ids().unwrap().iter().any(|p| p.bytes().into_iter().collect::<Vec<u8>>() == promise));
    // A second commit of a promise that is already running is refused.
    c.send(&commit_promise(8, promise, 0)).await;
    assert_eq!(c.recv_error().await.1, ErrorCode::NotFound);
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_query_result_pins_its_tier_0_items_for_the_session_until_it_ends() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut p, cam) = producer(&broker).await;
    let (arena, name, generation) = broker.arena();
    let loc = publish(&arena, &name, generation, 10);
    p.send(&ingest_located(0, cam, 1, 100, 200, &loc)).await;
    committed(&broker, &mut p).await;
    let continuum_core::StorageLocator::Shm { slab_class, slot, slot_generation, .. } = loc else { unreachable!() };
    let h = continuum_core::SlotHandle { class: slab_class, index: slot, generation: slot_generation };
    assert_eq!(arena.pin_count(h).unwrap(), 0);

    let mut c = consumer(&broker).await;
    c.send(&query_for(7, "cam0", 100, 200, fbs::QueryMode::HitOnly)).await;
    let (_, kind, bytes) = c.recv().await;
    assert_eq!(kind, ControlPayload::QueryResult);
    let r = read_envelope(&bytes).unwrap().payload_as_query_result().unwrap();
    assert_eq!(r.items().unwrap().get(0).locator().unwrap().tier(), continuum_core::StorageTier::Tier0Hot);
    assert_eq!(arena.pin_count(h).unwrap(), 1, "pinned for the reader");
    // A second query pins again: each result is the reader's own pin.
    c.send(&query_for(8, "cam0", 100, 200, fbs::QueryMode::HitOnly)).await;
    c.recv().await;
    assert_eq!(arena.pin_count(h).unwrap(), 2);
    // The session's end releases them.
    c.send(&goodbye()).await;
    assert!(c.closed().await);
    let released = async {
        while arena.pin_count(h).unwrap() != 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(5), released).await.expect("pins released when the session ends");
    // A slot that's gone stale isn't handed out as Tier 0.
    let mut c = consumer(&broker).await;
    arena.reclaim(h).unwrap();
    c.send(&query_for(9, "cam0", 100, 200, fbs::QueryMode::HitOnly)).await;
    let (_, kind, bytes) = c.recv().await;
    assert_eq!(kind, ControlPayload::QueryResult);
    let r = read_envelope(&bytes).unwrap().payload_as_query_result().unwrap();
    assert!(r.items().is_none_or(|i| i.is_empty()), "the only copy can't be pinned, so the item is left out");
    broker.stop().await;
}

// Subscriptions (core/subscribe.rs).

/// A `Subscribe` for `streams`, numbered `id` on the session.
struct Sub<'s> {
    id: u64,
    streams: &'s [&'s str],
    start: fbs::StartAt,
    start_ns: u64,
    durable: Option<&'s str>,
    pinned: bool,
    subjects: &'s [u8],
    shape: Option<&'s str>,
}

impl<'s> Sub<'s> {
    fn live(id: u64, streams: &'s [&'s str]) -> Self {
        Self {
            id,
            streams,
            start: fbs::StartAt::Live,
            start_ns: 0,
            durable: None,
            pinned: false,
            subjects: &[],
            shape: None,
        }
    }

    fn from_time(id: u64, streams: &'s [&'s str], start_ns: u64) -> Self {
        Self { start: fbs::StartAt::Time, start_ns, ..Self::live(id, streams) }
    }

    fn saved(id: u64, streams: &'s [&'s str], name: &'s str) -> Self {
        Self { start: fbs::StartAt::Saved, durable: Some(name), ..Self::live(id, streams) }
    }

    fn build(&self, request_id: u64) -> Vec<u8> {
        envelope(request_id, |b| {
            let names: Vec<_> = self.streams.iter().map(|s| b.create_string(s)).collect();
            let streams = b.create_vector(&names);
            let durable_name = self.durable.map(|d| b.create_string(d));
            let shape_name = self.shape.map(|s| b.create_string(s));
            let subjects = (!self.subjects.is_empty()).then(|| {
                let ids: Vec<_> = self
                    .subjects
                    .iter()
                    .map(|&s| {
                        let mut e = [0u8; 16];
                        e[6] = 0x70;
                        e[14] = 0xEE;
                        e[15] = s;
                        fbs::Uuid::new(&e)
                    })
                    .collect();
                b.create_vector(&ids)
            });
            let s = fbs::Subscribe::create(
                b,
                &fbs::SubscribeArgs {
                    subscription_id: self.id,
                    streams: Some(streams),
                    start: self.start,
                    start_ns: self.start_ns,
                    durable_name,
                    pinned: self.pinned,
                    subjects,
                    shape_name,
                },
            );
            (ControlPayload::Subscribe, s.as_union_value())
        })
    }
}

fn unsubscribe(request_id: u64, id: u64) -> Vec<u8> {
    envelope(request_id, |b| {
        let u = fbs::Unsubscribe::create(b, &fbs::UnsubscribeArgs { subscription_id: id });
        (ControlPayload::Unsubscribe, u.as_union_value())
    })
}

fn ack_events(id: u64, position: u64) -> Vec<u8> {
    envelope(0, |b| {
        let a = fbs::AckEvents::create(b, &fbs::AckEventsArgs { subscription_id: id, position });
        (ControlPayload::AckEvents, a.as_union_value())
    })
}

/// The next message, which must be a `Subscribed`: (request id, subscription id, position, live).
async fn subscribed(c: &mut Client) -> (u64, u64, u64, bool) {
    let (id, kind, bytes) = c.recv().await;
    assert_eq!(kind, ControlPayload::Subscribed, "expected a Subscribed");
    let s = read_envelope(&bytes).unwrap().payload_as_subscribed().unwrap();
    (id, s.subscription_id(), s.position(), s.live())
}

/// The next message, which must be a `StreamEvent` with request id 0: (subscription id, position, payload id's
/// last byte, whether it carries a locator).
async fn stream_event(c: &mut Client) -> (u64, u64, u8, bool) {
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (0, ControlPayload::StreamEvent), "expected a StreamEvent");
    let e = read_envelope(&bytes).unwrap().payload_as_stream_event().unwrap();
    let item = e.item().unwrap();
    let header = continuum_core::HeaderBuf::new(item.header_bytes().unwrap().bytes().to_vec()).unwrap();
    (e.subscription_id(), e.position(), header.id().0[15], item.locator().is_some())
}

/// The next message, which must be a `CaughtUp`: (subscription id, position).
async fn caught_up(c: &mut Client) -> (u64, u64) {
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (0, ControlPayload::CaughtUp), "expected a CaughtUp");
    let u = read_envelope(&bytes).unwrap().payload_as_caught_up().unwrap();
    (u.subscription_id(), u.position())
}

/// Payload ids and positions of the next `n` stream events, in the order they arrive.
async fn events(c: &mut Client, n: usize) -> Vec<(u8, u64)> {
    let mut got = Vec::with_capacity(n);
    for _ in 0..n {
        let (_, position, id, _) = stream_event(c).await;
        got.push((id, position));
    }
    got
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_live_subscription_gets_each_commit_in_order_until_it_unsubscribes() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut p, cam) = producer(&broker).await;
    // History before the subscription is not sent to a live subscriber.
    p.send(&ingest_copy(0, cam, 1, 100, 200, &[])).await;
    committed(&broker, &mut p).await;
    let mut c = consumer(&broker).await;
    c.send(&Sub::live(7, &["cam0"]).build(3)).await;
    let (id, sub, _, live) = subscribed(&mut c).await;
    assert_eq!((id, sub, live), (3, 7, true));

    for (n, start) in [(2u8, 200u64), (3, 300), (4, 400)] {
        p.send(&ingest_copy(0, cam, n, start, start + 100, &[])).await;
    }
    let got = events(&mut c, 3).await;
    assert_eq!(got.iter().map(|g| g.0).collect::<Vec<_>>(), vec![2, 3, 4]);
    assert!(got.windows(2).all(|w| w[0].1 < w[1].1), "positions increase: {got:?}");
    // Header only, since the subscription didn't ask for pinned data.
    c.send(&heartbeat(9)).await;
    assert_eq!(c.recv().await.1, ControlPayload::Heartbeat);

    c.send(&unsubscribe(4, 7)).await;
    let (id, kind, _) = c.recv().await;
    assert_eq!((id, kind), (4, ControlPayload::Unsubscribed));
    p.send(&ingest_copy(0, cam, 5, 500, 600, &[])).await;
    committed(&broker, &mut p).await;
    c.send(&heartbeat(10)).await;
    assert_eq!(c.recv().await.0, 10, "nothing arrived after the unsubscribe");
    // Unsubscribing twice, or a stream that doesn't exist, is refused.
    c.send(&unsubscribe(5, 7)).await;
    assert_eq!(c.recv_error().await.1, ErrorCode::NotFound);
    c.send(&Sub::live(8, &["nope"]).build(6)).await;
    assert_eq!(c.recv_error().await.1, ErrorCode::NotFound);
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_subscription_from_a_time_catches_up_then_goes_live() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut p, cam) = producer(&broker).await;
    for (n, start) in [(1u8, 100u64), (2, 200), (3, 300)] {
        p.send(&ingest_copy(0, cam, n, start, start + 100, &[])).await;
    }
    committed(&broker, &mut p).await;
    let mut c = consumer(&broker).await;
    // From 250: payload 2 (200 to 300) reaches into it, payload 1 doesn't.
    c.send(&Sub::from_time(1, &["cam0"], 250).build(3)).await;
    let (_, _, position, live) = subscribed(&mut c).await;
    assert_eq!((position, live), (0, false));
    assert_eq!(events(&mut c, 2).await.iter().map(|g| g.0).collect::<Vec<_>>(), vec![2, 3]);
    let (sub, caught) = caught_up(&mut c).await;
    assert_eq!(sub, 1);
    assert!(caught >= 3, "position {caught} covers the three commits");

    p.send(&ingest_copy(0, cam, 4, 400, 500, &[])).await;
    let (_, position, id, _) = stream_event(&mut c).await;
    assert_eq!(id, 4);
    assert!(position > caught);
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_durable_subscription_resumes_after_its_acknowledged_position() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut p, cam) = producer(&broker).await;
    for (n, start) in [(1u8, 100u64), (2, 200), (3, 300)] {
        p.send(&ingest_copy(0, cam, n, start, start + 100, &[])).await;
    }
    committed(&broker, &mut p).await;
    let mut c = consumer(&broker).await;
    // Nothing saved yet: everything.
    c.send(&Sub::saved(1, &["cam0"], "journal").build(3)).await;
    assert_eq!(subscribed(&mut c).await, (3, 1, 0, false));
    let got = events(&mut c, 3).await;
    assert_eq!(got.iter().map(|g| g.0).collect::<Vec<_>>(), vec![1, 2, 3]);
    caught_up(&mut c).await;
    // Handled through the second one; the cursor goes through the writer.
    c.send(&ack_events(1, got[1].1)).await;
    committed(&broker, &mut c).await;
    c.send(&unsubscribe(4, 1)).await;
    assert_eq!(c.recv().await.1, ControlPayload::Unsubscribed);

    // Another session (a restarted consumer) picks up after it.
    let mut d = consumer(&broker).await;
    d.send(&Sub::saved(5, &["cam0"], "journal").build(6)).await;
    assert_eq!(subscribed(&mut d).await, (6, 5, got[1].1, false));
    assert_eq!(events(&mut d, 1).await[0].0, 3);
    assert_eq!(caught_up(&mut d).await.0, 5);
    // A Saved start needs a durable name.
    let mut bad = Sub::saved(9, &["cam0"], "x");
    bad.durable = None;
    d.send(&bad.build(7)).await;
    assert_eq!(
        d.recv_error().await,
        (7, ErrorCode::BadRequest, "a Subscribe from a saved position needs a durable_name".into())
    );
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subscriptions_filter_by_subject_and_shape_and_carry_locators_when_pinned() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut p, cam) = producer(&broker).await;
    let mut c = consumer(&broker).await;
    let mut subjects = Sub::live(1, &["cam0"]);
    subjects.subjects = &[7];
    subjects.pinned = true;
    c.send(&subjects.build(3)).await;
    subscribed(&mut c).await;
    // The raw headers in these tests have no shape name, so a shape filter keeps nothing.
    let mut shaped = Sub::live(2, &["cam0"]);
    shaped.shape = Some("pose");
    c.send(&shaped.build(4)).await;
    subscribed(&mut c).await;
    // The same id twice on one session is a conflict.
    c.send(&Sub::live(2, &["cam0"]).build(5)).await;
    assert_eq!(c.recv_error().await.1, ErrorCode::Conflict);

    p.send(&ingest_copy(0, cam, 1, 100, 200, &[8])).await;
    p.send(&ingest_copy(0, cam, 2, 200, 300, &[7, 8])).await;
    p.send(&ingest_copy(0, cam, 3, 300, 400, &[])).await;
    let (sub, _, id, located) = stream_event(&mut c).await;
    assert_eq!((sub, id, located), (1, 2, true), "only the payload about subject 7, with its Tier 1 locator");
    c.send(&heartbeat(9)).await;
    assert_eq!(c.recv().await.1, ControlPayload::Heartbeat, "nothing else came through either filter");
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_subscription_over_two_streams_merges_them_in_commit_order_and_sees_lease_outputs() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut p, cam) = producer(&broker).await;
    let (mut w, cap, pose, _) = worker(&broker).await;
    let mut c = consumer(&broker).await;
    c.send(&Sub::live(1, &["cam0", "pose"]).build(3)).await;
    subscribed(&mut c).await;

    p.send(&ingest_copy(0, cam, 1, 100, 200, &[])).await;
    committed(&broker, &mut p).await;
    // A worker's committed output is published too.
    let lease = grant(&mut w, 5, cap, continuum_core::PriorityBand::P2Nearline).await;
    let spec =
        continuum_core::HeaderSpec::raw(continuum_core::Uuid(id_of(42)), continuum_core::StreamId(pose), 100, 200);
    let header = continuum_core::HeaderBuf::build(&spec).unwrap().into_bytes();
    let commit = envelope(6, |b| {
        let h = b.create_vector(&header);
        let loc = flatbuffers_locator(b, &chunk_at(continuum_core::StorageTier::Tier1Warm));
        let item = fbs::Item::create(b, &fbs::ItemArgs { locator: Some(loc), header_bytes: Some(h), commit_seq: 0 });
        let outputs = b.create_vector(&[item]);
        let c = fbs::CommitLease::create(
            b,
            &fbs::CommitLeaseArgs { lease_id: lease, outputs: Some(outputs), coverage: None, cost: None },
        );
        (ControlPayload::CommitLease, c.as_union_value())
    });
    w.send(&commit).await;
    acked(&mut w, 6, lease).await;
    p.send(&ingest_copy(0, cam, 2, 200, 300, &[])).await;
    let got = events(&mut c, 3).await;
    assert_eq!(got.iter().map(|g| g.0).collect::<Vec<_>>(), vec![1, 42, 2]);
    assert!(got.windows(2).all(|w| w[0].1 < w[1].1), "one increasing position across both streams: {got:?}");
    broker.stop().await;
}

/// An IngestNotice for payload `n` (two id bytes, for long histories) on `stream` over [start, end) with a Tier 1
/// copy and this shape name.
fn ingest_shaped(request_id: u64, stream: u32, n: u16, start: u64, end: u64, shape: &str) -> Vec<u8> {
    let mut id = [0u8; 16];
    id[6] = 0x70;
    id[14..16].copy_from_slice(&n.to_be_bytes());
    let mut spec =
        continuum_core::HeaderSpec::raw(continuum_core::Uuid(id), continuum_core::StreamId(stream), start, end);
    spec.shape_name = shape.to_owned();
    let header = continuum_core::HeaderBuf::build(&spec).unwrap().into_bytes();
    envelope(request_id, |b| {
        let h = b.create_vector(&header);
        let loc = flatbuffers_locator(b, &chunk_at(continuum_core::StorageTier::Tier1Warm));
        let n = IngestNotice::create(
            b,
            &IngestNoticeArgs { stream_id: stream, header_bytes: Some(h), locator: Some(loc), ..Default::default() },
        );
        (ControlPayload::IngestNotice, n.as_union_value())
    })
}

/// Stream events until the `CaughtUp`: (how many, the caught-up position).
async fn until_caught_up(c: &mut Client) -> (usize, u64) {
    let mut n = 0;
    loop {
        let (id, kind, bytes) = c.recv().await;
        assert_eq!(id, 0);
        match kind {
            ControlPayload::StreamEvent => n += 1,
            ControlPayload::CaughtUp => {
                return (n, read_envelope(&bytes).unwrap().payload_as_caught_up().unwrap().position());
            }
            other => panic!("unexpected {other:?}"),
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_over_more_than_one_page_skips_nothing_and_a_shape_filter_keeps_its_shape() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut p, cam) = producer(&broker).await;
    // 300 payloads, more than a catch-up page (256) holds, every other one shaped "pose".
    for n in 1u16..=300 {
        let start = 100 * u64::from(n);
        let shape = if n % 2 == 0 { "pose" } else { "" };
        p.send(&ingest_shaped(0, cam, n, start, start + 100, shape)).await;
        if n % 100 == 0 {
            committed(&broker, &mut p).await;
        }
    }
    let mut c = consumer(&broker).await;
    // From 27_050: payload 270 ([27000, 27100)) reaches it, and 271 to 300 start after it: 31 payloads, all past the
    // first page, which the time filter empties.
    c.send(&Sub::from_time(1, &["cam0"], 27_050).build(3)).await;
    assert_eq!(subscribed(&mut c).await, (3, 1, 0, false));
    let (n, caught) = until_caught_up(&mut c).await;
    assert_eq!(n, 31, "every payload reaching the start time is sent before CaughtUp");
    assert!(caught >= 300);
    // The shape filter keeps the 150 "pose" payloads out of the whole history.
    let mut shaped = Sub::from_time(2, &["cam0"], 1);
    shaped.shape = Some("pose");
    c.send(&shaped.build(4)).await;
    subscribed(&mut c).await;
    assert_eq!(until_caught_up(&mut c).await.0, 150);
    broker.stop().await;
}

// Purge notices to workers that keep their own copies (core/forget.rs).

/// A worker named `name` with no capabilities that keeps its own copies of payloads.
fn register_keeper(request_id: u64, name: &str) -> Vec<u8> {
    envelope(request_id, |b| {
        let caps = b.create_vector::<flatbuffers::WIPOffset<fbs::Capability>>(&[]);
        let name = b.create_string(name);
        let w = fbs::RegisterWorker::create(
            b,
            &fbs::RegisterWorkerArgs {
                worker_id: Some(name),
                version_hash: None,
                capabilities: Some(caps),
                device_index: None,
                keeps_copies: true,
            },
        );
        (ControlPayload::RegisterWorker, w.as_union_value())
    })
}

async fn keeper(broker: &Broker, name: &str) -> Client {
    let mut c = Client::connect(&broker.socket).await;
    c.send(&hello(1, SCHEMA_MAJOR)).await;
    assert_eq!(c.recv().await.1, ControlPayload::Welcome);
    c.send(&register_keeper(2, name)).await;
    assert_eq!(c.recv().await.1, ControlPayload::WorkerRegistered);
    c
}

fn purge_ack(purge_id: [u8; 16]) -> Vec<u8> {
    envelope(0, |b| {
        let id = fbs::Uuid::new(&purge_id);
        let a = fbs::PurgeAck::create(b, &fbs::PurgeAckArgs { purge_id: Some(&id) });
        (ControlPayload::PurgeAck, a.as_union_value())
    })
}

/// The next message, which must be a `PurgeResult`: (request id, purge id).
async fn purge_result_id(c: &mut Client) -> (u64, [u8; 16]) {
    let (id, kind, bytes) = c.recv().await;
    assert_eq!(kind, ControlPayload::PurgeResult, "expected a PurgeResult");
    let r = read_envelope(&bytes).unwrap().payload_as_purge_result().unwrap();
    (id, r.purge_id().unwrap().0)
}

/// The next message, which must be a `PurgeNotice` with request id 0: (purge id, payload ids' last bytes, spans as
/// (stream, start, end)). A single-page notice; see `purge_notice_pages` for a paged one.
async fn purge_notice(c: &mut Client) -> ([u8; 16], Vec<u8>, Vec<(u32, u64, u64)>) {
    let (id, ids, spans, more) = purge_notice_page(c).await;
    assert!(!more, "a single page");
    (id, ids, spans)
}

async fn purge_notice_page(c: &mut Client) -> ([u8; 16], Vec<u8>, Vec<(u32, u64, u64)>, bool) {
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (0, ControlPayload::PurgeNotice), "expected a PurgeNotice");
    let n = read_envelope(&bytes).unwrap().payload_as_purge_notice().unwrap();
    let mut ids: Vec<u8> = n.payload_ids().unwrap().iter().map(|u| u.0[15]).collect();
    ids.sort_unstable();
    let spans = n
        .spans()
        .unwrap()
        .iter()
        .map(|s| (s.stream_id(), s.range().unwrap().start_ns(), s.range().unwrap().end_ns()))
        .collect();
    (n.purge_id().unwrap().0, ids, spans, n.more())
}

/// Whether the purge's record says every copy is gone (after the writer has done what was queued).
async fn purge_completed(broker: &Broker, purge_id: [u8; 16]) -> bool {
    broker.core.sync_writer().await.unwrap();
    let (tx, rx) = oneshot::channel();
    let job: continuum_broker::core::ReadJob = Box::new(move |catalog| {
        let record = catalog.purge_record(continuum_core::Uuid(purge_id)).unwrap().expect("a purge record");
        let _ = tx.send(record.completed_ns.is_some());
    });
    broker.core.read(job).await.unwrap();
    rx.await.unwrap()
}

/// Polls for the purge's completion for up to 3 s.
async fn completes(broker: &Broker, purge_id: [u8; 16]) -> bool {
    for _ in 0..60 {
        if purge_completed(broker, purge_id).await {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    false
}

fn forget(grace_ms: u64, ack_timeout_ms: u64) -> continuum_broker::core::ForgetConfig {
    continuum_broker::core::ForgetConfig {
        keeper_grace: Duration::from_millis(grace_ms),
        ack_timeout: Duration::from_millis(ack_timeout_ms),
        ..Default::default()
    }
}

/// Ten payloads at 0, 10, .. 90 on cam0, then a purge of [20, 50): the purge's id.
async fn purged_three(broker: &Broker, c: &mut Client, cam: u32) -> [u8; 16] {
    for n in 0..10u8 {
        let t = u64::from(n) * 10;
        c.send(&ingest_copy(0, cam, n + 1, t, t + 10, &[])).await;
    }
    committed(broker, c).await;
    c.send(&purge_range(30, 20, 50, &["cam0"])).await;
    let (id, purge) = purge_result_id(c).await;
    assert_eq!(id, 30);
    purge
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_purge_with_no_copy_keepers_completes_once_the_grace_is_over() {
    let broker = Broker::start_forgetting(forget(1_000, 5_000)).await;
    let (mut c, cam) = producer(&broker).await;
    let purge = purged_three(&broker, &mut c, cam).await;
    assert!(!purge_completed(&broker, purge).await, "finished, but a keeper may still register during the grace");
    assert!(completes(&broker, purge).await);
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_purge_waits_for_its_keepers_ack_and_a_keeper_that_reconnects_is_told_again() {
    let broker = Broker::start_forgetting(forget(50, 10_000)).await;
    let mut k = keeper(&broker, "tier-worker").await;
    let (mut c, cam) = producer(&broker).await;
    let purge = purged_three(&broker, &mut c, cam).await;
    let (id, ids, spans) = purge_notice(&mut k).await;
    assert_eq!(id, purge);
    assert_eq!(ids, vec![3, 4, 5], "the payloads at 20, 30, and 40");
    assert_eq!(spans, vec![(cam, 20, 50)], "the steps' spans, merged");
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert!(!purge_completed(&broker, purge).await, "waiting for the keeper, grace or not");

    // The keeper goes away without acknowledging; the same worker, back on a new session, is told again.
    k.send(&goodbye()).await;
    assert!(k.closed().await);
    let mut k2 = keeper(&broker, "tier-worker").await;
    assert_eq!(purge_notice(&mut k2).await.0, purge);
    // Another keeper that registers now is told too, and must acknowledge as well.
    let mut v = keeper(&broker, "vector-worker").await;
    assert_eq!(purge_notice(&mut v).await.0, purge);
    k2.send(&purge_ack(purge)).await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!purge_completed(&broker, purge).await, "the vector worker hasn't answered");
    v.send(&purge_ack(purge)).await;
    assert!(completes(&broker, purge).await);
    // A keeper that acknowledged and re-registers isn't told again; an ack for an unknown purge is ignored.
    v.send(&goodbye()).await;
    assert!(v.closed().await);
    let mut v2 = keeper(&broker, "vector-worker").await;
    v2.send(&purge_ack([9; 16])).await;
    v2.send(&heartbeat(7)).await;
    assert_eq!(v2.recv().await, (7, ControlPayload::Heartbeat, v2.buf.clone()));
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_keeper_registering_during_the_grace_is_told_of_a_purge_that_finished_before_it() {
    let broker = Broker::start_forgetting(forget(2_000, 10_000)).await;
    let (mut c, cam) = producer(&broker).await;
    let purge = purged_three(&broker, &mut c, cam).await;
    let mut k = keeper(&broker, "tier-worker").await;
    assert_eq!(purge_notice(&mut k).await.0, purge);
    k.send(&purge_ack(purge)).await;
    assert!(completes(&broker, purge).await, "complete once the grace is over and the keeper has answered");
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_keeper_owed_more_notices_than_its_queue_holds_catches_up_instead_of_being_closed() {
    // On the Mac, the tier worker came back owed 1541 notices, all sent at once into a 1024-message
    // queue, so it was closed before it could acknowledge any, reconnected, and was closed again, forever.
    let broker = Broker::start_forgetting(forget(50, 60_000)).await;
    let mut k = keeper(&broker, "tier-worker").await;
    k.send(&goodbye()).await;
    assert!(k.closed().await);
    let (mut c, _cam) = producer(&broker).await;
    let owed = 1_300u64;
    for n in 0..owed {
        c.send(&purge_range(100 + n, n * 10, n * 10 + 5, &["cam0"])).await;
        purge_result_id(&mut c).await;
    }
    let mut k = keeper(&broker, "tier-worker").await;
    let mut seen = std::collections::HashSet::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    while (seen.len() as u64) < owed {
        assert!(tokio::time::Instant::now() < deadline, "only {} of {owed} notices arrived", seen.len());
        let (id, _, _) = purge_notice(&mut k).await;
        if seen.insert(id) {
            k.send(&purge_ack(id)).await;
        }
    }
    k.send(&heartbeat(7)).await;
    loop {
        let (rid, kind, _) = tokio::time::timeout(Duration::from_secs(5), k.recv()).await.expect("still connected");
        if kind == ControlPayload::Heartbeat {
            assert_eq!(rid, 7);
            break;
        }
    }
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_slow_keeper_is_sent_the_notice_again_after_the_ack_timeout() {
    let broker = Broker::start_forgetting(forget(50, 200)).await;
    let mut k = keeper(&broker, "tier-worker").await;
    let (mut c, cam) = producer(&broker).await;
    let purge = purged_three(&broker, &mut c, cam).await;
    assert_eq!(purge_notice(&mut k).await.0, purge);
    assert_eq!(purge_notice(&mut k).await.0, purge, "sent again after the timeout");
    k.send(&purge_ack(purge)).await;
    assert!(completes(&broker, purge).await);
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_purge_left_finished_by_a_previous_run_is_sent_to_a_keeper_at_the_next_start() {
    use continuum_catalog::Catalog as _;
    let (catalog, cam) = catalog_with_frames();
    let purge = continuum_core::Uuid([0x71; 16]);
    let selector = continuum_catalog::PurgeSelector::Range { span: Interval::new(20, 40).unwrap(), streams: None };
    catalog.start_purge(selector, "left over", "", purge, 1).unwrap();
    // Its rows went in the previous run, but no keeper acknowledged, so it is still pending.
    while !matches!(catalog.purge_step(purge, 1_000).unwrap(), continuum_catalog::PurgeProgress::Removed { .. }) {}
    assert_eq!(catalog.pending_purges().unwrap(), [purge]);
    let broker = Broker::start_cfg(Duration::from_secs(30), 16, Box::new(catalog), forget(2_000, 10_000)).await;
    let mut k = keeper(&broker, "tier-worker").await;
    let (id, ids, spans) = purge_notice(&mut k).await;
    assert_eq!(id, purge.0);
    assert!(!ids.is_empty(), "the report's removed ids");
    assert_eq!(spans, vec![(cam.0, 20, 40)], "the steps ran in the previous run, so the span comes from the record");
    k.send(&purge_ack(purge.0)).await;
    assert!(completes(&broker, purge.0).await);
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_second_keeper_under_a_connected_keepers_name_is_refused() {
    let broker = Broker::start_forgetting(forget(50, 10_000)).await;
    let k = keeper(&broker, "tier-worker").await;
    let mut twin = Client::connect(&broker.socket).await;
    twin.send(&hello(1, SCHEMA_MAJOR)).await;
    twin.recv().await;
    twin.send(&register_keeper(2, "tier-worker")).await;
    assert_eq!(twin.recv_error().await.1, ErrorCode::Conflict, "one acknowledgement per name, so one keeper per name");
    // A worker that keeps no copies may share the name; a keeper may take it once the first is gone.
    twin.send(&register_worker(3, "pose", "cam0")).await;
    assert_eq!(twin.recv().await.1, ControlPayload::WorkerRegistered, "a worker that keeps no copies may share it");
    drop(k);
    let mut again = Client::connect(&broker.socket).await;
    again.send(&hello(1, SCHEMA_MAJOR)).await;
    again.recv().await;
    for _ in 0..50 {
        again.send(&register_keeper(2, "tier-worker")).await;
        let (_, kind, _) = again.recv().await;
        if kind == ControlPayload::WorkerRegistered {
            break;
        }
        assert_eq!(kind, ControlPayload::Error, "still a conflict while the old session closes");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_large_purge_is_told_in_pages_and_acknowledged_once() {
    let mut cfg = forget(50, 10_000);
    cfg.notice_ids = 2;
    let broker = Broker::start_cfg(
        Duration::from_secs(30),
        continuum_broker::server::DEFAULT_MAX_SESSIONS,
        Box::new(continuum_catalog::MemoryCatalog::new()),
        cfg,
    )
    .await;
    let mut k = keeper(&broker, "tier-worker").await;
    let (mut c, cam) = producer(&broker).await;
    let purge = purged_three(&broker, &mut c, cam).await;
    let (id, first, spans, more) = purge_notice_page(&mut k).await;
    assert_eq!((id, first.len(), spans, more), (purge, 2, vec![(cam, 20, 50)], true), "two ids and the spans");
    let (id, second, spans, more) = purge_notice_page(&mut k).await;
    assert_eq!((id, second.len(), spans, more), (purge, 1, vec![], false), "the last id, no spans, last page");
    let mut all: Vec<u8> = first.into_iter().chain(second).collect();
    all.sort_unstable();
    assert_eq!(all, vec![3, 4, 5]);
    k.send(&purge_ack(purge)).await;
    assert!(completes(&broker, purge).await);
    broker.stop().await;
}

// ---- Rings carved at registration ----

/// A `RegisterStream` for a ring stream: `stride` byte records at `rate_hz`, kept `seconds`.
fn register_ring(request_id: u64, name: &str, rate_hz: f32, stride: u32, seconds: f32) -> Vec<u8> {
    envelope(request_id, |b| {
        let name = b.create_string(name);
        let shape = b.create_string("S24LE");
        let r = RegisterStream::create(
            b,
            &RegisterStreamArgs {
                name: Some(name),
                kind: StreamKind::Sensor,
                shape_name: Some(shape),
                shape_version: 1,
                nominal_rate_hz: rate_hz,
                ring_stride_bytes: stride,
                ring_seconds: seconds,
                ..Default::default()
            },
        );
        (ControlPayload::RegisterStream, r.as_union_value())
    })
}

/// The stream id, ring index, and ring layout a ring stream's `StreamRegistered` carries.
async fn ring_registered(c: &mut Client) -> (u32, u16, continuum_shm::CarvedRing, String, u32) {
    let (_, kind, bytes) = c.recv().await;
    assert_eq!(kind, ControlPayload::StreamRegistered, "{:?}", read_envelope(&bytes).unwrap().payload_as_error());
    let r = read_envelope(&bytes).unwrap().payload_as_stream_registered().unwrap();
    let l = r.lease().expect("a ring stream's lease carries its ring");
    let ring = continuum_shm::CarvedRing {
        ctrl_offset: l.ring_offset(),
        data_offset: l.ring_data_offset(),
        stride: l.ring_stride(),
        capacity: l.ring_records(),
        generation: l.ring_generation(),
    };
    (r.stream_id(), l.ring_index(), ring, l.arena().unwrap().to_owned(), l.arena_generation())
}

fn ring_ingest(request_id: u64, stream: u32, n: u8, span: (u64, u64), at: &continuum_core::StorageLocator) -> Vec<u8> {
    ingest_located(request_id, stream, n, span.0, span.1, at)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_ring_stream_gets_a_ring_of_exactly_its_stride_and_its_batches_ingest_as_tier_0() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;
    let (arena, name, generation) = welcome_arena(&mut c).await;
    // A 10 ms block of 48 kHz mono 24-bit PCM is 1,440 bytes, which no slot class would hold exactly.
    c.send(&register_ring(2, "mic0", 100.0, 1_440, 0.5)).await;
    let (mic, index, ring, ring_arena, ring_generation) = ring_registered(&mut c).await;
    assert_eq!((ring.stride, ring.capacity), (1_440, 50), "0.5 s at 100 blocks a second");
    assert_eq!((ring_arena.as_str(), ring_generation), (name.as_str(), generation));
    let mut w = arena.carved_ring_writer(&ring).unwrap();
    for n in 0..30u8 {
        w.push(&[n; 1_440]).unwrap();
    }
    let batch = |first_seq: u64, count: u32| continuum_core::StorageLocator::Ring {
        arena: name.clone(),
        arena_generation: generation,
        ring: index,
        first_seq,
        count,
    };
    // Request id 0: indexed silently (a numbered notice is answered with Ingested).
    c.send(&ring_ingest(0, mic, 1, (100, 200), &batch(0, 10))).await;
    c.send(&ring_ingest(0, mic, 2, (200, 300), &batch(10, 10))).await;
    committed(&broker, &mut c).await;
    let covered = broker.core.covered(continuum_core::StreamId(mic), Interval::new(0, 1_000).unwrap()).await.unwrap();
    assert_eq!(covered.len(), 1);
    assert_eq!(
        (covered[0].interval, covered[0].tier),
        (Interval::new(100, 300).unwrap(), continuum_core::StorageTier::Tier0Hot)
    );

    // Registering again (a capture worker that reconnected) gets the same ring back; another stride doesn't.
    c.send(&register_ring(5, "mic0", 100.0, 1_440, 0.5)).await;
    let (again, again_index, again_ring, _, _) = ring_registered(&mut c).await;
    assert_eq!((again, again_index, again_ring), (mic, index, ring));
    c.send(&register_ring(6, "mic0", 100.0, 3_840, 0.5)).await;
    let (id, code, message) = c.recv_error().await;
    assert_eq!((id, code), (6, ErrorCode::Unavailable));
    assert!(message.contains("1440 byte records"), "{message}");
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_ring_batch_the_ring_hasnt_written_or_has_lapped_is_rejected() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;
    let (arena, name, generation) = welcome_arena(&mut c).await;
    c.send(&register_ring(2, "mic0", 100.0, 1_440, 0.5)).await;
    let (mic, index, ring, _, _) = ring_registered(&mut c).await;
    let mut w = arena.carved_ring_writer(&ring).unwrap();
    for n in 0..80u8 {
        w.push(&[n; 1_440]).unwrap(); // 80 records in a ring of 50: 0 to 29 are lapped
    }
    let at = |arena_name: &str, arena_generation: u32, ring: u16, first_seq: u64, count: u32| {
        continuum_core::StorageLocator::Ring { arena: arena_name.into(), arena_generation, ring, first_seq, count }
    };
    let cases = [
        (at(&name, generation, index, 75, 10), "the batch ends at 85"),
        (at(&name, generation, index, 20, 10), "lapped"),
        (at(&name, generation, index + 1, 40, 10), "isn't stream"),
        (at(&name, generation + 1, index, 40, 10), "names arena"),
    ];
    for (i, (loc, why)) in cases.iter().enumerate() {
        let id = 10 + i as u64;
        c.send(&ring_ingest(id, mic, 1, (100, 200), loc)).await;
        let (got, reason) = rejected(&mut c).await;
        assert_eq!(got, id);
        assert!(reason.contains(why), "case {i}: {reason}");
    }
    // The newest 50 are good.
    c.send(&ring_ingest(0, mic, 1, (100, 200), &at(&name, generation, index, 30, 50))).await;
    committed(&broker, &mut c).await;
    let covered = broker.core.covered(continuum_core::StreamId(mic), Interval::new(0, 1_000).unwrap()).await.unwrap();
    assert_eq!(covered.len(), 1);
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_batch_the_ring_laps_leaves_the_index_and_a_ring_that_doesnt_fit_is_refused() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;
    let (arena, name, generation) = welcome_arena(&mut c).await;
    c.send(&register_ring(2, "mic0", 100.0, 1_440, 0.5)).await;
    let (mic, index, ring, _, _) = ring_registered(&mut c).await;
    let mut w = arena.carved_ring_writer(&ring).unwrap();
    for n in 0..10u8 {
        w.push(&[n; 1_440]).unwrap();
    }
    let loc = continuum_core::StorageLocator::Ring {
        arena: name.clone(),
        arena_generation: generation,
        ring: index,
        first_seq: 0,
        count: 10,
    };
    c.send(&ring_ingest(0, mic, 1, (100, 200), &loc)).await;
    committed(&broker, &mut c).await;
    let mic_stream = continuum_core::StreamId(mic);
    let window = Interval::new(0, 1_000).unwrap();
    assert_eq!(broker.core.covered(mic_stream, window).await.unwrap().len(), 1);
    // The ring laps the batch; within a tick or two the index lets it go.
    for n in 0..50u8 {
        w.push(&[n; 1_440]).unwrap();
    }
    let gone = async {
        while !broker.core.covered(mic_stream, window).await.unwrap().is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(5), gone).await.expect("the lapped batch leaves the index");

    // A second small ring fits; one that needs more than the region has left is refused with the reason.
    c.send(&register_ring(4, "imu0", 100.0, 64, 0.5)).await;
    ring_registered(&mut c).await;
    c.send(&register_ring(5, "mic1", 100.0, 1_440, 60.0)).await;
    let (id, code, message) = c.recv_error().await;
    assert_eq!((id, code), (5, ErrorCode::Unavailable));
    assert!(message.contains("ring region"), "{message}");
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_purged_ring_batch_lapped_later_leaves_an_overlapping_copy_alone() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;
    let (arena, name, generation) = welcome_arena(&mut c).await;
    c.send(&register_ring(2, "mic0", 100.0, 1_440, 0.5)).await;
    let (mic, index, ring, _, _) = ring_registered(&mut c).await;
    let mut w = arena.carved_ring_writer(&ring).unwrap();
    for n in 0..20u8 {
        w.push(&[n; 1_440]).unwrap();
    }
    let batch = |first_seq: u64| continuum_core::StorageLocator::Ring {
        arena: name.clone(),
        arena_generation: generation,
        ring: index,
        first_seq,
        count: 10,
    };
    // Batch 1 holds [100, 200); batch 2, later in the ring, holds [50, 250), over all of it.
    c.send(&ring_ingest(0, mic, 1, (100, 200), &batch(0))).await;
    c.send(&ring_ingest(0, mic, 2, (50, 250), &batch(10))).await;
    committed(&broker, &mut c).await;
    c.send(&purge_ids(3, &[1], "test", "")).await;
    purge_result_id(&mut c).await;
    let mic_stream = continuum_core::StreamId(mic);
    let window = Interval::new(0, 1_000).unwrap();
    let spans = || async {
        broker
            .core
            .covered(mic_stream, window)
            .await
            .unwrap()
            .iter()
            .map(|c| (c.interval.start_ns(), c.interval.end_ns()))
            .collect::<Vec<_>>()
    };
    assert_eq!(spans().await, [(50, 250)]);
    // The ring laps batch 1's records but not batch 2's, and a few sweeps pass. The purge already took batch 1's
    // copy, so batch 2's must stay whole.
    let lap = u64::from(ring.capacity) + 5 - 20;
    for _ in 0..lap {
        w.push(&[0; 1_440]).unwrap();
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(spans().await, [(50, 250)]);
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_purged_ring_batch_doesnt_come_back_as_the_unindexed_tail() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut p = Client::connect(&broker.socket).await;
    let (arena, name, generation) = welcome_arena(&mut p).await;
    p.send(&register_ring(2, "mic0", 100.0, 1_440, 0.5)).await;
    let (mic, index, ring, _, _) = ring_registered(&mut p).await;
    let mut w = arena.carved_ring_writer(&ring).unwrap();
    for n in 0..10u8 {
        w.push(&[n; 1_440]).unwrap();
    }
    let sealed = continuum_core::StorageLocator::Ring {
        arena: name,
        arena_generation: generation,
        ring: index,
        first_seq: 0,
        count: 10,
    };
    p.send(&ring_ingest(0, mic, 1, (1_000, 101_000), &sealed)).await;
    committed(&broker, &mut p).await;
    p.send(&purge_ids(3, &[1], "test", "")).await;
    purge_result_id(&mut p).await;
    // The purged records are still in the ring, but no query hands them out as its tail.
    let mut c = consumer(&broker).await;
    c.send(&query_for(7, "mic0", 1_000, 400_000, fbs::QueryMode::HitOnly)).await;
    let (_, kind, bytes) = c.recv().await;
    assert_eq!(kind, ControlPayload::QueryResult);
    let r = read_envelope(&bytes).unwrap().payload_as_query_result().unwrap();
    assert!(r.tail().is_none(), "the purged batch's records came back as the tail");
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_dead_writers_ring_can_be_taken_by_the_streams_next_writer() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;
    let (arena, _, _) = welcome_arena(&mut c).await;
    c.send(&register_ring(2, "mic0", 100.0, 1_440, 0.5)).await;
    let (_, _, ring, _, _) = ring_registered(&mut c).await;
    // The writer's process dies holding the ring (its guard is never dropped).
    std::mem::forget(arena.carved_ring_writer(&ring).unwrap());
    assert!(arena.carved_ring_writer(&ring).is_err(), "still claimed");
    c.send(&goodbye()).await;
    assert!(c.closed().await);
    let mut next = Client::connect(&broker.socket).await;
    let (arena, _, _) = welcome_arena(&mut next).await;
    let taken = async {
        loop {
            if let Ok(w) = arena.carved_ring_writer(&ring) {
                return w;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    let mut w = tokio::time::timeout(Duration::from_secs(5), taken).await.expect("released when its session ended");
    w.push(&[1; 1_440]).unwrap();
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_subscriber_learns_each_ring_streams_layout_and_can_follow_it() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut p = Client::connect(&broker.socket).await;
    let (arena, _, _) = welcome_arena(&mut p).await;
    p.send(&register_ring(2, "mic0", 100.0, 1_440, 0.5)).await;
    let (mic, index, ring, name, generation) = ring_registered(&mut p).await;
    p.send(&register_stream(3, "cam0", 30.0)).await;
    stream_registered(&mut p).await;

    let mut c = consumer(&broker).await;
    c.send(&Sub::live(1, &["mic0", "cam0"]).build(4)).await;
    let (_, kind, bytes) = c.recv().await;
    assert_eq!(kind, ControlPayload::Subscribed);
    let s = read_envelope(&bytes).unwrap().payload_as_subscribed().unwrap();
    let rings = s.rings().expect("the ring stream's layout");
    assert_eq!(rings.len(), 1, "only mic0 has a ring");
    let r = rings.get(0);
    let l = r.lease().unwrap();
    assert_eq!((r.stream_id(), l.ring_index()), (mic, index));
    assert_eq!((l.arena().unwrap(), l.arena_generation()), (name.as_str(), generation));
    let announced = continuum_shm::CarvedRing {
        ctrl_offset: l.ring_offset(),
        data_offset: l.ring_data_offset(),
        stride: l.ring_stride(),
        capacity: l.ring_records(),
        generation: l.ring_generation(),
    };
    assert_eq!(announced, ring);
    // A reader built from what the subscriber was told sees what the writer writes.
    let mut w = arena.carved_ring_writer(&ring).unwrap();
    let mut reader = arena.carved_ring_reader(&announced, continuum_shm::ReadFrom::Oldest).unwrap();
    w.push(&[7; 1_440]).unwrap();
    let mut out = [0u8; 1_440];
    assert_eq!(reader.read(&mut out).unwrap().map(|r| r.seq), Some(0));
    assert_eq!(out[0], 7);
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_ring_batch_committed_as_a_lease_output_leaves_the_index_once_lapped() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut w = Client::connect(&broker.socket).await;
    let (arena, name, generation) = welcome_arena(&mut w).await;
    w.send(&register_stream(2, "cam0", 30.0)).await;
    stream_registered(&mut w).await;
    // A worker whose output is a ring stream (a resampler, say).
    w.send(&register_ring(3, "pose", 100.0, 1_440, 0.5)).await;
    let (pose, index, ring, _, _) = ring_registered(&mut w).await;
    w.send(&register_worker(4, "pose", "cam0")).await;
    let (_, kind, bytes) = w.recv().await;
    assert_eq!(kind, ControlPayload::WorkerRegistered);
    let cap = read_envelope(&bytes).unwrap().payload_as_worker_registered().unwrap().capability_ids().unwrap().get(0);
    let lease = grant(&mut w, 5, cap, continuum_core::PriorityBand::P2Nearline).await;
    let mut writer = arena.carved_ring_writer(&ring).unwrap();
    for n in 0..10u8 {
        writer.push(&[n; 1_440]).unwrap();
    }
    let loc = continuum_core::StorageLocator::Ring {
        arena: name,
        arena_generation: generation,
        ring: index,
        first_seq: 0,
        count: 10,
    };
    let mut id = [0u8; 16];
    id[6] = 0x70;
    id[15] = 9;
    let spec = continuum_core::HeaderSpec::raw(continuum_core::Uuid(id), continuum_core::StreamId(pose), 100, 200);
    let header = continuum_core::HeaderBuf::build(&spec).unwrap().into_bytes();
    let commit = envelope(6, |b| {
        let h = b.create_vector(&header);
        let l = flatbuffers_locator(b, &loc);
        let item = fbs::Item::create(b, &fbs::ItemArgs { locator: Some(l), header_bytes: Some(h), commit_seq: 0 });
        let outputs = b.create_vector(&[item]);
        let c = fbs::CommitLease::create(
            b,
            &fbs::CommitLeaseArgs { lease_id: lease, outputs: Some(outputs), coverage: None, cost: None },
        );
        (ControlPayload::CommitLease, c.as_union_value())
    });
    w.send(&commit).await;
    acked(&mut w, 6, lease).await;
    let stream = continuum_core::StreamId(pose);
    let window = Interval::new(0, 1_000).unwrap();
    assert_eq!(broker.core.covered(stream, window).await.unwrap().len(), 1);
    for n in 0..50u8 {
        writer.push(&[n; 1_440]).unwrap();
    }
    let gone = async {
        while !broker.core.covered(stream, window).await.unwrap().is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(5), gone).await.expect("the lapped output leaves the index");
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_ring_batch_in_another_streams_ring_is_rejected() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;
    let (arena, name, generation) = welcome_arena(&mut c).await;
    c.send(&register_ring(2, "mic0", 100.0, 1_440, 0.5)).await;
    let (_, mic_ring, ring, _, _) = ring_registered(&mut c).await;
    c.send(&register_ring(3, "mic1", 100.0, 64, 0.5)).await;
    let (mic1, _, _, _, _) = ring_registered(&mut c).await;
    arena.carved_ring_writer(&ring).unwrap().push(&[1; 1_440]).unwrap();
    // mic1's header, naming mic0's ring.
    let loc = continuum_core::StorageLocator::Ring {
        arena: name,
        arena_generation: generation,
        ring: mic_ring,
        first_seq: 0,
        count: 1,
    };
    c.send(&ring_ingest(4, mic1, 1, (100, 200), &loc)).await;
    let (id, reason) = rejected(&mut c).await;
    assert_eq!(id, 4);
    assert!(reason.contains("isn't stream"), "{reason}");
    broker.stop().await;
}

#[tokio::test]
async fn a_producer_ingests_silently_and_a_numbered_notice_or_repeat_gets_its_sequence() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = Client::connect(&broker.socket).await;
    c.send(&hello(1, SCHEMA_MAJOR)).await;
    c.recv().await;
    c.send(&register_stream(2, "cam0", 30.0)).await;
    let (_, cam) = stream_registered(&mut c).await;
    c.send(&ingest(0, cam, &header(1, cam))).await;
    // The normal path (request id 0) has no reply: the next thing back answers the heartbeat.
    c.send(&heartbeat(4)).await;
    assert_eq!(c.recv().await.1, ControlPayload::Heartbeat);
    // A numbered notice is answered with the payload's commit sequence once it is committed, and a repeat with the
    // same header bytes is idempotent: the same sequence, no rejection.
    c.send(&ingest(5, cam, &header_at(2, cam, 200))).await;
    let (id, stream, payload, seq) = ingested(&mut c).await;
    assert_eq!((id, stream, payload), (5, cam, 2));
    assert!(seq > 0);
    c.send(&ingest(6, cam, &header_at(2, cam, 200))).await;
    assert_eq!(ingested(&mut c).await, (6, cam, 2, seq), "the existing sequence");
    c.send(&ingest(7, cam, &header(1, cam))).await;
    assert_eq!(ingested(&mut c).await.3, seq - 1, "committed before payload 2");
    // The same id with a different header is a conflict.
    c.send(&ingest(8, cam, &header_at(2, cam, 500))).await;
    let (id, reason, code) = rejected_with_code(&mut c).await;
    assert_eq!((id, code), (8, ErrorCode::Conflict));
    assert!(!reason.is_empty());
    broker.stop().await;
}

/// The next message, which must be an `Ingested`: (request id, stream id, payload id's last byte, commit sequence).
async fn ingested(c: &mut Client) -> (u64, u32, u8, u64) {
    let (id, kind, bytes) = c.recv().await;
    assert_eq!(kind, ControlPayload::Ingested, "expected an Ingested");
    let i = read_envelope(&bytes).unwrap().payload_as_ingested().unwrap();
    (id, i.stream_id(), i.payload_id().unwrap().0[15], i.commit_seq())
}

fn header_at(n: u8, stream: u32, start: u64) -> Vec<u8> {
    let mut id = [0u8; 16];
    id[6] = 0x70;
    id[15] = n;
    let spec =
        continuum_core::HeaderSpec::raw(continuum_core::Uuid(id), continuum_core::StreamId(stream), start, start);
    continuum_core::HeaderBuf::build(&spec).unwrap().into_bytes()
}

/// An IngestNotice for payload `n` whose bytes travel inline (a client with no shared memory).
fn ingest_inline(request_id: u64, stream: u32, n: u8, start: u64, bytes: &[u8]) -> Vec<u8> {
    let spec = continuum_core::HeaderSpec::raw(
        continuum_core::Uuid(id_of(n)),
        continuum_core::StreamId(stream),
        start,
        start + 10,
    );
    let header = continuum_core::HeaderBuf::build(&spec).unwrap().into_bytes();
    let locator = continuum_core::StorageLocator::Inline { bytes: bytes.to_vec() };
    envelope(request_id, |b| {
        let h = b.create_vector(&header);
        let loc = locator.build_into(b);
        let n = IngestNotice::create(
            b,
            &IngestNoticeArgs { stream_id: stream, header_bytes: Some(h), locator: Some(loc), ..Default::default() },
        );
        (ControlPayload::IngestNotice, n.as_union_value())
    })
}

/// The next message, which must be a `QueryResult`: (payload ids' last bytes with their commit sequences, complete,
/// next_after).
async fn seq_result(c: &mut Client, request_id: u64) -> (Vec<(u8, u64)>, bool, u64) {
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (request_id, ControlPayload::QueryResult));
    let r = read_envelope(&bytes).unwrap().payload_as_query_result().unwrap();
    let items = r
        .items()
        .unwrap()
        .iter()
        .map(|it| {
            let header = continuum_core::HeaderBuf::new(it.header_bytes().unwrap().bytes().to_vec()).unwrap();
            (header.id().0[15], it.commit_seq())
        })
        .collect();
    (items, r.complete(), r.next_after())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_inline_payload_is_stored_and_served_back_inline_and_an_oversized_one_is_rejected() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut p, cam) = producer(&broker).await;
    p.send(&ingest_inline(5, cam, 1, 100, b"a small event")).await;
    let seq = ingested(&mut p).await.3;
    assert!(seq > 0);
    p.send(&ingest_inline(6, cam, 2, 200, &[7u8; continuum_core::MAX_INLINE_BYTES + 1])).await;
    let (id, reason, code) = rejected_with_code(&mut p).await;
    assert_eq!((id, code), (6, ErrorCode::BadRequest));
    assert!(reason.contains("inline"), "{reason}");

    let mut c = consumer(&broker).await;
    c.send(&query_for(7, "cam0", 0, 1_000, fbs::QueryMode::HitOnly)).await;
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (7, ControlPayload::QueryResult));
    let r = read_envelope(&bytes).unwrap().payload_as_query_result().unwrap();
    let item = r.items().unwrap().get(0);
    assert_eq!(item.commit_seq(), seq, "rows carry their commit sequence");
    let locator = continuum_core::StorageLocator::from_table(&item.locator().unwrap()).unwrap();
    assert_eq!(locator, continuum_core::StorageLocator::Inline { bytes: b"a small event".to_vec() });
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_sequence_query_pages_in_commit_order_and_filters_by_subject_id_and_time() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut p, cam) = producer(&broker).await;
    // Five payloads; 2 and 4 are about subject 7.
    for n in 1..=5u8 {
        let t = u64::from(n) * 100;
        let subjects: &[u8] = if n % 2 == 0 { &[7] } else { &[] };
        p.send(&ingest_copy(0, cam, n, t, t + 50, subjects)).await;
    }
    committed(&broker, &mut p).await;
    let mut c = consumer(&broker).await;

    // Two at a time, continuing from next_after.
    let mut q = SeqQuery::after("cam0", 0);
    q.max_items = 2;
    c.send(&q.build(10)).await;
    let (items, complete, next) = seq_result(&mut c, 10).await;
    assert_eq!(items.iter().map(|i| i.0).collect::<Vec<_>>(), [1, 2]);
    assert!(items.windows(2).all(|w| w[0].1 < w[1].1), "commit order: {items:?}");
    assert!(!complete && next == items[1].1, "continue after the last row scanned");
    let mut q = SeqQuery::after("cam0", next);
    q.max_items = 10;
    c.send(&q.build(11)).await;
    let (items, complete, next) = seq_result(&mut c, 11).await;
    assert_eq!((items.iter().map(|i| i.0).collect::<Vec<_>>(), complete, next), (vec![3, 4, 5], true, 0));

    // Filters: subjects, ids, start_before, and a range.
    let mut q = SeqQuery::after("cam0", 0);
    q.subjects = &[7];
    c.send(&q.build(12)).await;
    assert_eq!(seq_result(&mut c, 12).await.0.iter().map(|i| i.0).collect::<Vec<_>>(), [2, 4]);
    let mut q = SeqQuery::after("cam0", 0);
    q.ids = &[5, 1];
    c.send(&q.build(13)).await;
    assert_eq!(seq_result(&mut c, 13).await.0.iter().map(|i| i.0).collect::<Vec<_>>(), [1, 5]);
    let mut q = SeqQuery::after("cam0", 0);
    q.start_before = 250;
    c.send(&q.build(14)).await;
    assert_eq!(seq_result(&mut c, 14).await.0.iter().map(|i| i.0).collect::<Vec<_>>(), [1, 2]);
    let mut q = SeqQuery::after("cam0", 0);
    q.range = Some((320, 420));
    c.send(&q.build(15)).await;
    assert_eq!(seq_result(&mut c, 15).await.0.iter().map(|i| i.0).collect::<Vec<_>>(), [3, 4]);
    // A sequence query makes nothing, so a mode that would quote is refused; an unknown stream is not found.
    let mut q = SeqQuery::after("cam0", 0);
    q.mode = fbs::QueryMode::QuoteIfMissing;
    c.send(&q.build(16)).await;
    assert_eq!(c.recv_error().await.1, ErrorCode::BadRequest);
    c.send(&SeqQuery::after("nope", 0).build(17)).await;
    assert_eq!(c.recv_error().await.1, ErrorCode::NotFound);
    broker.stop().await;
}

/// A sequence `Query`.
struct SeqQuery<'s> {
    stream: &'s str,
    after_seq: u64,
    max_items: u32,
    max_scan: u32,
    subjects: &'s [u8],
    ids: &'s [u8],
    start_before: u64,
    range: Option<(u64, u64)>,
    mode: fbs::QueryMode,
}

impl<'s> SeqQuery<'s> {
    fn after(stream: &'s str, after_seq: u64) -> Self {
        Self {
            stream,
            after_seq,
            max_items: 0,
            max_scan: 0,
            subjects: &[],
            ids: &[],
            start_before: 0,
            range: None,
            mode: fbs::QueryMode::HitOnly,
        }
    }

    fn build(&self, request_id: u64) -> Vec<u8> {
        envelope(request_id, |b| {
            let name = b.create_string(self.stream);
            let uuids = |b: &mut FlatBufferBuilder<'static>, ns: &[u8], tag: u8| {
                (!ns.is_empty()).then(|| {
                    let v: Vec<_> = ns
                        .iter()
                        .map(|&n| {
                            let mut e = [0u8; 16];
                            e[6] = 0x70;
                            e[14] = tag;
                            e[15] = n;
                            fbs::Uuid::new(&e)
                        })
                        .collect();
                    b.create_vector(&v)
                })
            };
            let subjects = uuids(b, self.subjects, 0xEE);
            let ids = uuids(b, self.ids, 0);
            let range = self.range.map(|(s, e)| fbs::TimeRange::new(s, e));
            let q = Query::create(
                b,
                &QueryArgs {
                    stream: Some(name),
                    range: range.as_ref(),
                    band: continuum_core::PriorityBand::P2Nearline,
                    mode: self.mode,
                    max_items: self.max_items,
                    by_seq: true,
                    after_seq: self.after_seq,
                    subjects,
                    start_before: self.start_before,
                    ids,
                    max_scan: self.max_scan,
                    ..Default::default()
                },
            );
            (ControlPayload::Query, q.as_union_value())
        })
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_repeat_reaches_a_subscriber_once_and_a_ranged_page_reads_only_the_window() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut p, cam) = producer(&broker).await;
    let mut c = consumer(&broker).await;
    c.send(&Sub::live(1, &["cam0"]).build(3)).await;
    subscribed(&mut c).await;
    for n in 1..=5u8 {
        let t = u64::from(n) * 100;
        p.send(&ingest_copy(0, cam, n, t, t + 50, &[])).await;
    }
    committed(&broker, &mut p).await;
    // The same five again, byte for byte: acknowledged with their sequences, published to nobody.
    for n in 1..=5u8 {
        let t = u64::from(n) * 100;
        p.send(&ingest_copy(u64::from(n), cam, n, t, t + 50, &[])).await;
        assert!(ingested(&mut p).await.3 > 0);
    }
    assert_eq!(events(&mut c, 5).await.iter().map(|e| e.0).collect::<Vec<_>>(), [1, 2, 3, 4, 5]);
    c.send(&heartbeat(9)).await;
    assert_eq!(c.recv().await.1, ControlPayload::Heartbeat, "the repeats weren't published again");

    // One row a page over a range only payloads 3 and 4 overlap: the catalog reads the window from its time index,
    // so payloads 1 and 2 cost no pages at all.
    let mut got = Vec::new();
    let mut empty_pages = 0;
    let mut after = 0;
    for request in 20.. {
        let mut q = SeqQuery::after("cam0", after);
        q.max_items = 1;
        q.range = Some((320, 420));
        c.send(&q.build(request)).await;
        let (items, complete, next) = seq_result(&mut c, request).await;
        if items.is_empty() {
            empty_pages += 1;
        }
        got.extend(items.iter().map(|i| i.0));
        if complete {
            assert_eq!(next, 0);
            break;
        }
        assert!(next > after, "the position moves on");
        after = next;
    }
    assert_eq!(got, [3, 4]);
    assert_eq!(empty_pages, 0, "payloads 1 and 2 don't reach the range, so no page is spent on them");
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_commits_coverage_must_cite_one_of_the_leases_inputs() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut w, cap, pose, cam) = worker(&broker).await;
    let lease = grant(&mut w, 5, cap, continuum_core::PriorityBand::P2Nearline).await;
    // pose's input is cam0; coverage over pose itself isn't something this lease ran over.
    w.send(&commit_with(6, lease, &output_on(pose, 1, 100), Some(pose))).await;
    let (id, code, message) = w.recv_error().await;
    assert_eq!((id, code), (6, ErrorCode::NotAllowed));
    assert!(message.contains("isn't one of the lease's inputs"), "{message}");
    // Refused before anything was stored, so the lease is still the worker's: citing cam0 goes through.
    w.send(&commit_with(7, lease, &output_on(pose, 1, 100), Some(cam))).await;
    acked(&mut w, 7, lease).await;
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_worker_killed_mid_job_fails_the_promise_it_served() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (w, mut c, promise, _dispatch, _) = dispatched(&broker).await;
    // The worker dies holding the job: no Goodbye, the socket just closes.
    drop(w);
    // No other worker produces pose, so the promise can't be replanned: the consumer hears it failed.
    let (id, kind, bytes) = tokio::time::timeout(Duration::from_secs(5), c.recv()).await.expect("told within 5 s");
    assert_eq!((id, kind), (0, ControlPayload::PromiseFailed));
    let f = read_envelope(&bytes).unwrap().payload_as_promise_failed().unwrap();
    assert_eq!(f.promise_id().unwrap().bytes().into_iter().collect::<Vec<u8>>(), promise);
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_query_reaching_now_on_a_ring_stream_returns_the_unsealed_tail_too() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut p = Client::connect(&broker.socket).await;
    let (arena, name, generation) = welcome_arena(&mut p).await;
    // 10 ms PCM blocks, indexed one batch of 10 at a time.
    p.send(&register_ring(2, "mic0", 100.0, 1_440, 0.5)).await;
    let (mic, index, ring, _, _) = ring_registered(&mut p).await;
    let mut w = arena.carved_ring_writer(&ring).unwrap();
    for n in 0..13u8 {
        w.push(&[n; 1_440]).unwrap();
    }
    let sealed = continuum_core::StorageLocator::Ring {
        arena: name.clone(),
        arena_generation: generation,
        ring: index,
        first_seq: 0,
        count: 10,
    };
    p.send(&ring_ingest(0, mic, 1, (1_000, 101_000), &sealed)).await;
    committed(&broker, &mut p).await;

    // "The last 300 ms": past the sealed batch, so the three records written since come as the tail.
    let mut c = consumer(&broker).await;
    c.send(&query_for(7, "mic0", 1_000, 400_000, fbs::QueryMode::HitOnly)).await;
    let (_, kind, bytes) = c.recv().await;
    assert_eq!(kind, ControlPayload::QueryResult);
    let r = read_envelope(&bytes).unwrap().payload_as_query_result().unwrap();
    assert_eq!(r.items().map_or(0, |i| i.len()), 1, "the sealed batch");
    let tail = r.tail().expect("the unsealed tail");
    assert_eq!((tail.first_seq(), tail.last_seq()), (10, 12));
    assert_eq!((tail.ring(), tail.arena().unwrap(), tail.arena_generation()), (index, name.as_str(), generation));
    assert_eq!(tail.ring_offset(), ring.ctrl_offset);
    assert!(!r.complete(), "the tail isn't indexed yet");
    // `missing` still names the span the tail covers: `complete` and the tail are what tell a reader that those
    // bytes are readable now rather than gone (docs/CONTINUUM-BROKER.md, "the live edge").
    let missing: Vec<(u64, u64)> =
        r.missing().map(|m| m.iter().map(|t| (t.start_ns(), t.end_ns())).collect()).unwrap_or_default();
    assert_eq!(missing, vec![(101_000, 400_000)]);
    // A reader can copy the tail's records straight from the ring.
    let mut reader = arena.carved_ring_reader(&ring, continuum_shm::ReadFrom::Oldest).unwrap();
    let mut out = [0u8; 1_440];
    let mut seqs = Vec::new();
    while let Some(rec) = reader.read(&mut out).unwrap() {
        if rec.seq >= tail.first_seq() {
            assert_eq!(u64::from(out[0]), rec.seq);
            seqs.push(rec.seq);
        }
    }
    assert_eq!(seqs, [10, 11, 12]);

    // A query that stops inside what's indexed has no tail.
    c.send(&query_for(8, "mic0", 1_000, 50_000, fbs::QueryMode::HitOnly)).await;
    let (_, _, bytes) = c.recv().await;
    assert!(read_envelope(&bytes).unwrap().payload_as_query_result().unwrap().tail().is_none());
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_time_query_on_a_ring_nothing_has_been_written_to_answers_with_no_tail() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut p = Client::connect(&broker.socket).await;
    let _ = welcome_arena(&mut p).await;
    p.send(&register_ring(2, "mic0", 100.0, 1_440, 0.5)).await;
    let _ = ring_registered(&mut p).await;
    let mut c = consumer(&broker).await;
    c.send(&query_for(7, "mic0", 1_000, 400_000, fbs::QueryMode::HitOnly)).await;
    let (_, kind, bytes) = c.recv().await;
    assert_eq!(kind, ControlPayload::QueryResult);
    let r = read_envelope(&bytes).unwrap().payload_as_query_result().unwrap();
    assert!(r.tail().is_none(), "an empty ring has no tail");
    assert_eq!(r.items().map_or(0, |i| i.len()), 0);
    broker.stop().await;
}

// Purge by ids and the purge listing.

fn purge_ids(request_id: u64, ids: &[u8], reason: &str, detail: &str) -> Vec<u8> {
    envelope(request_id, |b| {
        let ids: Vec<_> = ids.iter().map(|&n| fbs::Uuid::new(&id_of(n))).collect();
        let ids = b.create_vector(&ids);
        let reason = b.create_string(reason);
        let detail = b.create_string(detail);
        let p = Purge::create(
            b,
            &PurgeArgs {
                range: None,
                streams: None,
                entity_id: None,
                reason: Some(reason),
                ids: Some(ids),
                detail: Some(detail),
                age_out: false,
                committed_before_ns: 0,
                aged_gap_ns: 0,
            },
        );
        (ControlPayload::Purge, p.as_union_value())
    })
}

fn list_purges(request_id: u64, after: Option<[u8; 16]>, limit: u32) -> Vec<u8> {
    envelope(request_id, |b| {
        let after = after.map(|a| fbs::Uuid::new(&a));
        let l = fbs::ListPurges::create(b, &fbs::ListPurgesArgs { after: after.as_ref(), limit });
        (ControlPayload::ListPurges, l.as_union_value())
    })
}

fn get_purge(request_id: u64, purge: [u8; 16]) -> Vec<u8> {
    envelope(request_id, |b| {
        let id = fbs::Uuid::new(&purge);
        let g = fbs::GetPurge::create(b, &fbs::GetPurgeArgs { purge_id: Some(&id) });
        (ControlPayload::GetPurge, g.as_union_value())
    })
}

/// A listed purge: (id, kind, id_count, reason, detail, removed, finished, completed_ns).
type Listed = ([u8; 16], fbs::PurgeKind, u32, String, String, u64, bool, u64);

fn listed(s: fbs::PurgeSummary<'_>) -> Listed {
    (
        s.purge_id().unwrap().0,
        s.kind(),
        s.id_count(),
        s.reason().unwrap_or("").to_owned(),
        s.detail().unwrap_or("").to_owned(),
        s.removed(),
        s.finished(),
        s.completed_ns(),
    )
}

/// The next message, which must be a `PurgeList`.
async fn purge_list(c: &mut Client, request_id: u64) -> Vec<Listed> {
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (request_id, ControlPayload::PurgeList));
    let l = read_envelope(&bytes).unwrap().payload_as_purge_list().unwrap();
    l.purges().unwrap().iter().map(listed).collect()
}

/// The next message, which must be a `PurgeRecord`: (summary, removed payloads summed over the counts, frames).
async fn purge_record(c: &mut Client, request_id: u64) -> (Listed, u64, u64) {
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (request_id, ControlPayload::PurgeRecord));
    let r = read_envelope(&bytes).unwrap().payload_as_purge_record().unwrap();
    let removed = r.removed().map_or(0, |v| v.iter().map(|x| x.count()).sum());
    (listed(r.summary().unwrap()), removed, r.removed_frames())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_purge_by_ids_removes_those_payloads_and_its_record_keeps_the_reason_and_detail() {
    let broker = Broker::start_forgetting(forget(100, 10_000)).await;
    let (mut c, cam) = producer(&broker).await;
    for n in 1..=5u8 {
        let t = u64::from(n) * 10;
        c.send(&ingest_copy(0, cam, n, t, t + 10, &[])).await;
    }
    committed(&broker, &mut c).await;
    c.send(&purge_ids(30, &[2, 4], "owner asked", r#"{"source":"test"}"#)).await;
    let (id, purge) = purge_result_id(&mut c).await;
    assert_eq!(id, 30);
    assert_eq!(covered_ns(&broker, cam).await, [(10, 20), (30, 40), (50, 60)], "2 and 4 are gone, the rest stay");

    c.send(&list_purges(31, None, 0)).await;
    let purges = purge_list(&mut c, 31).await;
    assert_eq!(purges.len(), 1);
    let (listed_id, kind, count, reason, detail, removed, finished, _) = &purges[0];
    assert_eq!(
        (*listed_id, *kind, *count, reason.as_str(), detail.as_str(), *removed, *finished),
        (purge, fbs::PurgeKind::Ids, 2, "owner asked", r#"{"source":"test"}"#, 2, true)
    );
    c.send(&get_purge(32, purge)).await;
    let (summary, removed, frames) = purge_record(&mut c, 32).await;
    assert_eq!((summary.0, summary.1, removed, frames), (purge, fbs::PurgeKind::Ids, 2, 0));
    // Complete once the grace is over (no keeper holds copies): the listing says when.
    assert!(completes(&broker, purge).await);
    c.send(&get_purge(33, purge)).await;
    assert!(purge_record(&mut c, 33).await.0.7 > 0, "completed_ns is set");
    // An unknown purge, and a Purge naming both ids and a range.
    c.send(&get_purge(34, [9; 16])).await;
    assert_eq!(c.recv_error().await.1, ErrorCode::NotFound);
    let both = envelope(35, |b| {
        let ids = b.create_vector(&[fbs::Uuid::new(&id_of(1))]);
        let range = fbs::TimeRange::new(0, 10);
        let p = Purge::create(b, &PurgeArgs { range: Some(&range), ids: Some(ids), ..Default::default() });
        (ControlPayload::Purge, p.as_union_value())
    });
    c.send(&both).await;
    assert_eq!(c.recv_error().await.1, ErrorCode::BadRequest);
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_purge_listing_pages_oldest_first() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut c, cam) = producer(&broker).await;
    for n in 1..=3u8 {
        let t = u64::from(n) * 10;
        c.send(&ingest_copy(0, cam, n, t, t + 10, &[])).await;
    }
    committed(&broker, &mut c).await;
    let mut ids = Vec::new();
    for n in 1..=3u8 {
        c.send(&purge_ids(40 + u64::from(n), &[n], "one", "")).await;
        ids.push(purge_result_id(&mut c).await.1);
    }
    c.send(&list_purges(50, None, 2)).await;
    let first = purge_list(&mut c, 50).await;
    assert_eq!(first.iter().map(|p| p.0).collect::<Vec<_>>(), ids[..2]);
    c.send(&list_purges(51, Some(first[1].0), 2)).await;
    let rest = purge_list(&mut c, 51).await;
    assert_eq!(rest.iter().map(|p| p.0).collect::<Vec<_>>(), ids[2..]);
    c.send(&list_purges(52, Some([9; 16]), 2)).await;
    assert_eq!(c.recv_error().await.1, ErrorCode::NotFound, "after a purge that isn't one");
    broker.stop().await;
}

/// Opens the redb catalog at `path` once the broker that had it has let go (its core thread exits just after stop).
async fn reopen(path: &Path) -> continuum_catalog::RedbCatalog {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        match continuum_catalog::RedbCatalog::open(path) {
            Ok(c) => return c,
            Err(e) => {
                assert!(std::time::Instant::now() < deadline, "the catalog stays held: {e}");
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cost_models_and_clock_samples_are_saved_and_the_next_run_starts_from_them() {
    use continuum_catalog::Catalog as _;
    let keep = tempfile::tempdir().unwrap();
    let file = keep.path().join("catalog.redb");
    let catalog = Box::new(continuum_catalog::RedbCatalog::open(&file).unwrap());
    let broker =
        Broker::start_on(Duration::from_secs(30), continuum_broker::server::DEFAULT_MAX_SESSIONS, catalog).await;
    let (mut w, cap, pose, _) = worker(&broker).await;
    let lease = grant(&mut w, 5, cap, continuum_core::PriorityBand::P2Nearline).await;
    // The job measured 3 ms for one item.
    let header = output_on(pose, 1, 100);
    let commit = envelope(6, |b| {
        let h = b.create_vector(&header);
        let item = fbs::Item::create(b, &fbs::ItemArgs { locator: None, header_bytes: Some(h), commit_seq: 0 });
        let outputs = b.create_vector(&[item]);
        let cost =
            fbs::ComputeCost::create(b, &fbs::ComputeCostArgs { wall_time_us: 3_000, items: 1, ..Default::default() });
        let c = fbs::CommitLease::create(
            b,
            &fbs::CommitLeaseArgs { lease_id: lease, outputs: Some(outputs), coverage: None, cost: Some(cost) },
        );
        (ControlPayload::CommitLease, c.as_union_value())
    });
    w.send(&commit).await;
    acked(&mut w, 6, lease).await;
    let learned = broker.core.cost_models().await.unwrap();
    let model = learned.iter().find(|m| m.samples > 0).expect("the commit taught the scheduler");
    assert!((model.ema_us_per_item - 3_000.0).abs() < 1e-6, "{model:?}");
    broker.core.upkeep_now().await.unwrap();
    broker.core.sync_writer().await.unwrap();
    drop(w);
    broker.stop().await;

    // The catalog has both: the cost model, and a clock point for this boot near now.
    let catalog = reopen(&file).await;
    let saved = continuum_broker::costs::decode(&catalog.load_cost_models().unwrap().expect("saved")).unwrap();
    assert!(saved.iter().any(|m| m.key == model.key && m.samples == 1));
    let boot = continuum_broker::host::boot_id().unwrap();
    let near = catalog.clock_near(boot, continuum_broker::host::monotonic_ns()).unwrap();
    let point = near.at_or_before.expect("a sample from this run");
    assert!(point.utc_ns.abs_diff(continuum_broker::host::utc_ns()) < 60_000_000_000, "{point:?}");

    // The next run starts from what was learned, not from the manifest's guess.
    let broker =
        Broker::start_on(Duration::from_secs(30), continuum_broker::server::DEFAULT_MAX_SESSIONS, Box::new(catalog))
            .await;
    let restored = broker.core.cost_models().await.unwrap();
    assert!(restored.iter().any(|m| m.key == model.key && m.samples == 1), "{restored:?}");
    broker.stop().await;
}

// Stream listing.

fn list_streams(request_id: u64) -> Vec<u8> {
    envelope(request_id, |b| {
        let l = fbs::ListStreams::create(b, &fbs::ListStreamsArgs {});
        (ControlPayload::ListStreams, l.as_union_value())
    })
}

/// The next message, which must be a `StreamList`: (id, name, kind, live) per stream.
async fn stream_list(c: &mut Client, request_id: u64) -> Vec<(u32, String, fbs::StreamKind, bool)> {
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (request_id, ControlPayload::StreamList));
    let l = read_envelope(&bytes).unwrap().payload_as_stream_list().unwrap();
    l.streams()
        .unwrap()
        .iter()
        .map(|s| (s.stream_id(), s.name().unwrap_or("").to_owned(), s.kind(), s.live()))
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_stream_listing_names_every_stream_and_says_which_have_a_producer() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (p, cam) = producer(&broker).await;
    let (mut w, _, pose, _) = worker(&broker).await;
    let mut c = consumer(&broker).await;
    c.send(&list_streams(5)).await;
    let listed = stream_list(&mut c, 5).await;
    assert_eq!(
        listed,
        [
            (cam, "cam0".to_owned(), fbs::StreamKind::Sensor, true),
            (pose, "pose".to_owned(), fbs::StreamKind::Derived, true),
        ],
        "in id order, both with a producer connected"
    );
    // The worker (which registered pose) leaves: pose stays listed, no longer live.
    w.send(&goodbye()).await;
    assert!(w.closed().await);
    assert!(wait_for(&broker, |live| !live.contains(&continuum_core::StreamId(pose))).await);
    c.send(&list_streams(6)).await;
    let listed = stream_list(&mut c, 6).await;
    assert_eq!(listed.iter().map(|s| (s.0, s.3)).collect::<Vec<_>>(), [(cam, true), (pose, false)]);
    drop(p);
    broker.stop().await;
}

/// Waits until `check` holds, for up to 5 s.
async fn eventually(what: &str, mut check: impl AsyncFnMut() -> bool) {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while !check().await {
        assert!(std::time::Instant::now() < deadline, "{what}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_purge_zeroes_its_payloads_tier_0_slots() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut p = Client::connect(&broker.socket).await;
    let (arena, name, generation) = welcome_arena(&mut p).await;
    p.send(&register_stream(2, "cam0", 30.0)).await;
    let (_, cam) = stream_registered(&mut p).await;
    let mut slot = arena.claim(0, 1).unwrap();
    slot.as_mut_slice()[..100].fill(0xAB);
    let offset = slot.offset() as u64;
    let h = slot.publish(100).unwrap();
    let loc = continuum_core::StorageLocator::Shm {
        arena: name,
        arena_generation: generation,
        slab_class: h.class,
        slot: h.index,
        slot_generation: h.generation,
        offset,
        length: 100,
    };
    p.send(&ingest_located(0, cam, 1, 100, 200, &loc)).await;
    committed(&broker, &mut p).await;
    p.send(&purge_range(3, 0, 1_000, &["cam0"])).await;
    assert_eq!(purge_result(&mut p).await.1, 1);
    eventually("the purged slot is zeroed and freed", async || {
        arena.scan(0).unwrap().iter().any(|s| {
            s.handle.index == h.index
                && s.handle.generation != h.generation
                && s.state == continuum_shm::SlotState::Free
        })
    })
    .await;
    // SAFETY: the slot lies inside the mapping, and nothing writes it now.
    let bytes = unsafe { std::slice::from_raw_parts(arena.base_ptr().add(offset as usize), 100) };
    assert!(bytes.iter().all(|&b| b == 0), "the purged payload's bytes are gone");
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_class_running_low_has_its_flushed_then_oldest_slots_reclaimed_and_the_index_follows() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut p = Client::connect(&broker.socket).await;
    let (arena, name, generation) = welcome_arena(&mut p).await;
    p.send(&register_stream(2, "cam0", 30.0)).await;
    let (_, cam) = stream_registered(&mut p).await;
    // Fill class 0's four slots, each a frame of its own, waiting for each to be indexed before the next: the broker
    // only reclaims slots it has indexed. The class is full from the fourth publish on, and a scan can run before
    // that frame is indexed, so the flushed frame is the third, not the last: every scan of a full class sees it.
    let mut handles = Vec::new();
    for n in 0..4u8 {
        let mut slot = arena.claim(0, 1).unwrap();
        slot.as_mut_slice()[0] = n;
        // This frame already has a Tier 1 copy, so it goes first, ahead of the two older unflushed ones. Marked before
        // it's published, so the broker's scan never sees it unflushed.
        if n == 2 {
            arena.mark_flushed(slot.handle()).unwrap();
        }
        let offset = slot.offset() as u64;
        let h = slot.publish(1).unwrap();
        let loc = continuum_core::StorageLocator::Shm {
            arena: name.clone(),
            arena_generation: generation,
            slab_class: h.class,
            slot: h.index,
            slot_generation: h.generation,
            offset,
            length: 1,
        };
        let start = 100 * u64::from(n) + 100;
        let request = 10 + u64::from(n);
        p.send(&ingest_located(request, cam, n + 1, start, start + 50, &loc)).await;
        let (accepted_request, _, accepted_payload, _) = ingested(&mut p).await;
        assert_eq!((accepted_request, accepted_payload), (request, n + 1));
        handles.push(h);
    }
    committed(&broker, &mut p).await;
    let stream = continuum_core::StreamId(cam);
    let all = Interval::new(0, 10_000).unwrap();
    // No free slots of four: within a scan or two the broker frees one, the flushed one.
    eventually("a slot is reclaimed", async || {
        arena.scan(0).unwrap().iter().any(|s| s.state == continuum_shm::SlotState::Free)
    })
    .await;
    let freed: Vec<_> =
        arena.scan(0).unwrap().into_iter().filter(|s| s.state == continuum_shm::SlotState::Free).collect();
    assert_eq!(freed.len(), 1, "back up to a quarter free: {freed:?}");
    assert_eq!(freed[0].handle.index, handles[2].index, "the flushed slot went first");
    eventually("the index lets go of the reclaimed copy, and only that one", async || {
        let covered = broker.core.covered(stream, all).await.unwrap();
        covered.iter().map(|c| (c.interval.start_ns(), c.interval.end_ns())).collect::<Vec<_>>()
            == [(100, 150), (200, 250), (400, 450)]
    })
    .await;
    broker.stop().await;
}

/// How a stream's slots are followed in [`full_unflushed_class`].
#[derive(Clone, Copy, PartialEq)]
enum Follower {
    /// A worker that keeps copies follows it durably, as the tier worker does: its slots are owed to the archive.
    Keeper,
    /// A durable subscription from a session that keeps no copies: nothing is owed to the archive.
    Reader,
}

/// Class 1 (two slots) full of `cam1` payloads with no archive copy and no reader pin, and class 0 (four slots) full
/// of flushed `cam0` payloads, so a scheduler scan that frees a class-0 slot proves it also looked at class 1.
/// Returns the arena, the class-1 handles, `cam1`'s id, and the follower's session (kept open).
async fn full_unflushed_class(
    broker: &Broker,
    p: &mut Client,
    follower: Follower,
) -> (continuum_shm::Arena, Vec<continuum_shm::SlotHandle>, continuum_core::StreamId, Client) {
    let (arena, name, generation) = welcome_arena(p).await;
    p.send(&register_stream(2, "cam0", 30.0)).await;
    let (_, cam) = stream_registered(p).await;
    p.send(&register_stream(3, "cam1", 30.0)).await;
    let (_, unflushed_cam) = stream_registered(p).await;
    let mut follow = match follower {
        Follower::Keeper => keeper(broker, "tier-under-test").await,
        Follower::Reader => {
            let mut c = Client::connect(&broker.socket).await;
            c.send(&hello(1, SCHEMA_MAJOR)).await;
            assert_eq!(c.recv().await.1, ControlPayload::Welcome);
            c
        }
    };
    follow.send(&Sub::saved(1, &["cam1"], "follower/cam1").build(3)).await;
    subscribed(&mut follow).await;
    let mut unflushed = Vec::new();
    for (class, count, stream) in [(1, 2, unflushed_cam), (0, 4, cam)] {
        for n in 0..count {
            let mut slot = arena.claim(class, 1).unwrap();
            slot.as_mut_slice()[0] = n;
            if class == 0 {
                arena.mark_flushed(slot.handle()).unwrap();
            }
            let offset = slot.offset() as u64;
            let h = slot.publish(1).unwrap();
            let loc = continuum_core::StorageLocator::Shm {
                arena: name.clone(),
                arena_generation: generation,
                slab_class: h.class,
                slot: h.index,
                slot_generation: h.generation,
                offset,
                length: 1,
            };
            let start = 100 * u64::from(n) + 100;
            let payload = n + 1 + u8::try_from(class).unwrap() * 10;
            let request = 100 + u64::from(payload);
            p.send(&ingest_located(request, stream, payload, start, start + 50, &loc)).await;
            let (accepted_request, accepted_stream, accepted_payload, _) = ingested(p).await;
            assert_eq!((accepted_request, accepted_stream, accepted_payload), (request, stream, payload));
            if class == 1 {
                unflushed.push(h);
            }
        }
    }
    committed(broker, p).await;
    eventually("the flushed sentinel proves a scheduler scan ran", async || {
        arena.scan(0).unwrap().iter().any(|s| s.state == continuum_shm::SlotState::Free)
    })
    .await;
    (arena, unflushed, continuum_core::StreamId(unflushed_cam), follow)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_full_class_keeps_unflushed_unpinned_payloads_until_they_are_flushed() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut p = Client::connect(&broker.socket).await;
    let (arena, unflushed, stream, _tier) = full_unflushed_class(&broker, &mut p, Follower::Keeper).await;
    // This core RPC is processed after the scan that freed the class-0 sentinel,
    // including its class-1 work and index updates.
    let all = Interval::new(0, 10_000).unwrap();
    let covered = broker.core.covered(stream, all).await.unwrap();
    assert_eq!(arena.used(1).unwrap(), 2, "unarchived slots must survive scheduler pressure");
    assert_eq!(covered.len(), 2, "both unarchived payloads remain indexed");
    for (n, h) in unflushed.iter().enumerate() {
        assert_eq!(arena.pin_count(*h).unwrap(), 0);
        let lease = arena.lease(*h, Duration::from_secs(1)).expect("unarchived payload remains readable");
        let mut bytes = [255];
        lease.copy_to(&mut bytes).unwrap();
        assert_eq!(bytes, [n as u8]);
        lease.release();
    }
    arena.mark_flushed(unflushed[0]).unwrap();
    eventually("a newly flushed slot becomes reclaimable", async || {
        arena
            .scan(1)
            .unwrap()
            .iter()
            .any(|s| s.handle.index == unflushed[0].index && s.state == continuum_shm::SlotState::Free)
    })
    .await;
    eventually("only the flushed payload leaves the index", async || {
        let remaining = broker.core.covered(stream, all).await.unwrap();
        remaining.len() == 1 && remaining[0].interval == Interval::new(200, 250).unwrap()
    })
    .await;
    let lease = arena.lease(unflushed[1], Duration::from_secs(1)).unwrap();
    let mut bytes = [255];
    lease.copy_to(&mut bytes).unwrap();
    assert_eq!(bytes, [1]);
    lease.release();
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_full_class_of_a_stream_nothing_archives_still_frees_its_oldest_unflushed_slot() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut p = Client::connect(&broker.socket).await;
    // A durable reader that keeps no copies owes nothing to the archive, so it can't starve the class.
    let (arena, unflushed, stream, _reader) = full_unflushed_class(&broker, &mut p, Follower::Reader).await;
    eventually("the oldest unflushed slot is reclaimed under pressure", async || {
        arena
            .scan(1)
            .unwrap()
            .iter()
            .any(|s| s.handle.index == unflushed[0].index && s.state == continuum_shm::SlotState::Free)
    })
    .await;
    assert_eq!(arena.used(1).unwrap(), 1, "an eighth of two slots, at least one, is reclaimed");
    let all = Interval::new(0, 10_000).unwrap();
    eventually("the reclaimed payload leaves the index", async || {
        let remaining = broker.core.covered(stream, all).await.unwrap();
        remaining.len() == 1 && remaining[0].interval == Interval::new(200, 250).unwrap()
    })
    .await;
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_purge_that_only_cut_frames_is_re_sent_after_a_restart_with_its_range_as_the_spans() {
    use continuum_catalog::{Catalog as _, ChunkFrame, NewChunk, NewPayload};
    // A committed chunk holding three frames at 10, 20, and 30.
    let catalog = continuum_catalog::MemoryCatalog::new();
    let cam = catalog
        .register_stream(continuum_catalog::StreamInfo {
            name: "cam0".into(),
            kind: continuum_catalog::StreamKind::Sensor,
            layer: continuum_core::DomainLayer::Layer1Raw,
            format: continuum_core::PayloadFormat::RawBytes,
            shape_name: String::new(),
            shape_version: 0,
            rate_hz: None,
            anchor: None,
        })
        .unwrap();
    let frame = |n: u8, t: u64| NewPayload {
        header: continuum_core::HeaderBuf::build(&continuum_core::HeaderSpec::raw(
            continuum_core::Uuid(id_of(n)),
            cam,
            t,
            t,
        ))
        .unwrap(),
        location: Some(continuum_core::StorageLocator::Shm {
            arena: "a".into(),
            arena_generation: 1,
            slab_class: 0,
            slot: u32::from(n),
            slot_generation: 1,
            offset: 0,
            length: 1,
        }),
        coverage: None,
    };
    catalog.insert(vec![frame(1, 10), frame(2, 20), frame(3, 30)]).unwrap();
    let chunk = NewChunk {
        payload: NewPayload {
            header: continuum_core::HeaderBuf::build(&continuum_core::HeaderSpec::raw(
                continuum_core::Uuid(id_of(100)),
                cam,
                10,
                31,
            ))
            .unwrap(),
            location: Some(chunk_at(continuum_core::StorageTier::Tier1Warm)),
            coverage: None,
        },
        frames: [10u64, 20, 30].map(|t| ChunkFrame::new(t, t * 10, 300)).to_vec(),
    };
    catalog.commit_chunk(chunk, &[id_of(1), id_of(2), id_of(3)].map(continuum_core::Uuid)).unwrap();
    let purge = continuum_core::Uuid([0x72; 16]);
    // Inside the chunk: the purge cuts a frame and removes no payload (only the absorbed frame's row, which every
    // absorbed frame keeps), so its report lists nothing removed. It ran to the end in the previous run; no keeper
    // acknowledged, so it is pending at this start.
    let selector = continuum_catalog::PurgeSelector::Range { span: Interval::new(20, 25).unwrap(), streams: None };
    catalog.start_purge(selector, "left over", "", purge, 1).unwrap();
    let report = loop {
        if let continuum_catalog::PurgeProgress::Removed { report, .. } = catalog.purge_step(purge, 1_000).unwrap() {
            break report;
        }
    };
    assert!(report.removed.is_empty() && !report.partial_chunks.is_empty(), "{report:?}");
    let broker = Broker::start_cfg(Duration::from_secs(30), 16, Box::new(catalog), forget(2_000, 10_000)).await;
    let mut k = keeper(&broker, "tier-worker").await;
    let (id, ids, spans) = purge_notice(&mut k).await;
    assert_eq!(id, purge.0);
    assert_eq!(ids, vec![2], "no payload removed whole: only the cut frame's row, named for the keepers");
    assert_eq!(spans, vec![(cam.0, 20, 25)], "the range on every stream the broker knows, so Tier 1 cuts it");
    k.send(&purge_ack(purge.0)).await;
    assert!(completes(&broker, purge.0).await);
    broker.stop().await;
}

fn forget_keeper(request_id: u64, name: &str) -> Vec<u8> {
    envelope(request_id, |b| {
        let name = b.create_string(name);
        let f = fbs::ForgetKeeper::create(b, &fbs::ForgetKeeperArgs { name: Some(name) });
        (ControlPayload::ForgetKeeper, f.as_union_value())
    })
}

/// A `KeeperForgotten`: (name, released).
async fn keeper_forgotten(c: &mut Client, request_id: u64) -> (String, u32) {
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (request_id, ControlPayload::KeeperForgotten), "expected a KeeperForgotten");
    let k = read_envelope(&bytes).unwrap().payload_as_keeper_forgotten().unwrap();
    (k.name().unwrap_or_default().to_owned(), k.released())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_keeper_told_before_a_restart_is_told_again_and_the_purge_waits_for_it_past_the_grace() {
    use continuum_catalog::Catalog as _;
    let keep = tempfile::tempdir().unwrap();
    let file = keep.path().join("catalog.redb");
    let catalog = Box::new(continuum_catalog::RedbCatalog::open(&file).unwrap());
    let broker = Broker::start_cfg(Duration::from_secs(30), 16, catalog, forget(50, 10_000)).await;
    let mut k = keeper(&broker, "tier-worker").await;
    let (mut c, cam) = producer(&broker).await;
    let purge = purged_three(&broker, &mut c, cam).await;
    assert_eq!(purge_notice(&mut k).await.0, purge);
    // The broker stops before the keeper answers. The catalog remembers the keeper, and the purge is still pending.
    broker.stop().await;
    let catalog = reopen(&file).await;
    assert_eq!(catalog.keepers().unwrap(), ["tier-worker"]);
    assert_eq!(catalog.pending_purges().unwrap(), [continuum_core::Uuid(purge)]);
    let broker = Broker::start_cfg(Duration::from_secs(30), 16, Box::new(catalog), forget(50, 10_000)).await;
    tokio::time::sleep(Duration::from_millis(250)).await;
    assert!(!purge_completed(&broker, purge).await, "past the grace, still waiting for the keeper it knows");
    let mut k = keeper(&broker, "tier-worker").await;
    let (id, ids, _) = purge_notice(&mut k).await;
    assert_eq!(id, purge);
    assert_eq!(ids, vec![3, 4, 5], "the report's removed ids, as before the restart");
    k.send(&purge_ack(purge)).await;
    assert!(completes(&broker, purge).await);
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_keeper_that_acknowledged_before_a_restart_is_not_told_again_and_the_purge_completes_without_it() {
    use continuum_catalog::Catalog as _;
    let keep = tempfile::tempdir().unwrap();
    let file = keep.path().join("catalog.redb");
    let catalog = Box::new(continuum_catalog::RedbCatalog::open(&file).unwrap());
    // A long grace: the purge is acknowledged but not complete when the broker stops.
    let broker = Broker::start_cfg(Duration::from_secs(30), 16, catalog, forget(30_000, 10_000)).await;
    let mut k = keeper(&broker, "tier-worker").await;
    let (mut c, cam) = producer(&broker).await;
    let purge = purged_three(&broker, &mut c, cam).await;
    assert_eq!(purge_notice(&mut k).await.0, purge);
    k.send(&purge_ack(purge)).await;
    // The ack reaches the core and the catalog before the stop, which would drop one still on the socket.
    committed(&broker, &mut k).await;
    assert!(!purge_completed(&broker, purge).await, "the grace is long");
    broker.stop().await;
    let catalog = reopen(&file).await;
    assert_eq!(catalog.purge_record(continuum_core::Uuid(purge)).unwrap().unwrap().acked, ["tier-worker"]);
    let broker = Broker::start_cfg(Duration::from_secs(30), 16, Box::new(catalog), forget(50, 10_000)).await;
    assert!(completes(&broker, purge).await, "the only keeper answered in the previous run");
    // The keeper registering now isn't told of it.
    let mut k = keeper(&broker, "tier-worker").await;
    k.send(&heartbeat(7)).await;
    assert_eq!(k.recv().await, (7, ControlPayload::Heartbeat, k.buf.clone()));
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_purge_waits_past_the_grace_for_a_keeper_the_broker_knows_but_that_isnt_connected() {
    let broker = Broker::start_forgetting(forget(50, 10_000)).await;
    // The keeper registered once and is between runs when the purge finishes.
    let mut k = keeper(&broker, "tier-worker").await;
    k.send(&goodbye()).await;
    assert!(k.closed().await);
    let (mut c, cam) = producer(&broker).await;
    let purge = purged_three(&broker, &mut c, cam).await;
    tokio::time::sleep(Duration::from_millis(250)).await;
    assert!(!purge_completed(&broker, purge).await, "the keeper is known, so the grace passing isn't enough");
    let mut k = keeper(&broker, "tier-worker").await;
    assert_eq!(purge_notice(&mut k).await.0, purge);
    k.send(&purge_ack(purge)).await;
    assert!(completes(&broker, purge).await);
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn forgetting_a_keeper_releases_the_purges_that_waited_on_it() {
    use continuum_catalog::Catalog as _;
    let keep = tempfile::tempdir().unwrap();
    let file = keep.path().join("catalog.redb");
    let catalog = Box::new(continuum_catalog::RedbCatalog::open(&file).unwrap());
    let broker = Broker::start_cfg(Duration::from_secs(30), 16, catalog, forget(50, 10_000)).await;
    let mut k = keeper(&broker, "old-tier-worker").await;
    k.send(&goodbye()).await;
    assert!(k.closed().await);
    let (mut c, cam) = producer(&broker).await;
    let purge = purged_three(&broker, &mut c, cam).await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!purge_completed(&broker, purge).await, "waiting for the retired worker");
    // The worker was renamed: an admin drops the old name, and the purge no longer waits for it.
    c.send(&forget_keeper(40, "old-tier-worker")).await;
    assert_eq!(keeper_forgotten(&mut c, 40).await, ("old-tier-worker".to_owned(), 1));
    assert!(completes(&broker, purge).await);
    c.send(&forget_keeper(41, "old-tier-worker")).await;
    assert_eq!(keeper_forgotten(&mut c, 41).await, ("old-tier-worker".to_owned(), 0), "gone already");
    c.send(&forget_keeper(42, "")).await;
    assert_eq!(c.recv_error().await.1, ErrorCode::BadRequest);
    // A connected keeper's name can't be dropped from under it.
    let _k = keeper(&broker, "tier-worker").await;
    c.send(&forget_keeper(43, "tier-worker")).await;
    assert_eq!(c.recv_error().await.1, ErrorCode::Conflict);
    // Dropped in the catalog too: the next run doesn't wait for it either (the connected one is remembered).
    broker.stop().await;
    let catalog = reopen(&file).await;
    assert_eq!(catalog.keepers().unwrap(), ["tier-worker"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_time_query_refuses_the_sequence_filters_instead_of_ignoring_them() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut c, _cam) = producer(&broker).await;
    let filtered = envelope(5, |b| {
        let name = b.create_string("cam0");
        let subject = fbs::Uuid::new(&[0xEE; 16]);
        let subjects = b.create_vector(&[subject]);
        let range = fbs::TimeRange::new(0, 100);
        let q = Query::create(
            b,
            &QueryArgs {
                stream: Some(name),
                range: Some(&range),
                mode: fbs::QueryMode::HitOnly,
                subjects: Some(subjects),
                ..Default::default()
            },
        );
        (ControlPayload::Query, q.as_union_value())
    });
    c.send(&filtered).await;
    let (id, code, message) = c.recv_error().await;
    assert_eq!((id, code), (5, ErrorCode::BadRequest));
    assert!(message.contains("subjects") && message.contains("by_seq"), "{message}");
    let scanned = envelope(6, |b| {
        let name = b.create_string("cam0");
        let range = fbs::TimeRange::new(0, 100);
        let q = Query::create(
            b,
            &QueryArgs {
                stream: Some(name),
                range: Some(&range),
                mode: fbs::QueryMode::HitOnly,
                max_scan: 10,
                ..Default::default()
            },
        );
        (ControlPayload::Query, q.as_union_value())
    });
    c.send(&scanned).await;
    assert_eq!(c.recv_error().await.1, ErrorCode::BadRequest, "max_scan too");
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_scan_cap_given_without_an_item_cap_bounds_the_page() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut c, cam) = producer(&broker).await;
    for n in 1..=5u8 {
        let t = u64::from(n) * 10;
        c.send(&ingest_copy(0, cam, n, t, t + 10, &[])).await;
    }
    committed(&broker, &mut c).await;
    let mut q = SeqQuery::after("cam0", 0);
    q.max_scan = 2;
    c.send(&q.build(7)).await;
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (7, ControlPayload::QueryResult));
    let r = read_envelope(&bytes).unwrap().payload_as_query_result().unwrap();
    assert_eq!(r.items().map_or(0, |i| i.len()), 2, "two rows scanned, so two items at most");
    assert!(r.next_after() > 0, "three remain");
    // An item cap the caller did set still wins over a smaller scan cap.
    let mut q = SeqQuery::after("cam0", r.next_after());
    q.max_scan = 1;
    q.max_items = 3;
    c.send(&q.build(8)).await;
    let (_, _, bytes) = c.recv().await;
    let r = read_envelope(&bytes).unwrap().payload_as_query_result().unwrap();
    assert_eq!((r.items().map_or(0, |i| i.len()), r.next_after()), (3, 0), "the rest, in one page");
    broker.stop().await;
}

fn remember_forgotten(request_id: u64, hash: [u8; 32]) -> Vec<u8> {
    envelope(request_id, |b| {
        let hash = fbs::ForgottenHash::new(&hash);
        let r = fbs::RememberForgotten::create(b, &fbs::RememberForgottenArgs { hash: Some(&hash) });
        (ControlPayload::RememberForgotten, r.as_union_value())
    })
}

fn is_forgotten(request_id: u64, hash: Option<[u8; 32]>) -> Vec<u8> {
    envelope(request_id, |b| {
        let hash = hash.map(|h| fbs::ForgottenHash::new(&h));
        let r = fbs::IsForgotten::create(b, &fbs::IsForgottenArgs { hash: hash.as_ref() });
        (ControlPayload::IsForgotten, r.as_union_value())
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_forgotten_names_hash_is_remembered_and_recognized() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let mut c = consumer(&broker).await;
    c.send(&is_forgotten(5, Some([7; 32]))).await;
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (5, ControlPayload::ForgottenStatus));
    assert!(!read_envelope(&bytes).unwrap().payload_as_forgotten_status().unwrap().forgotten(), "not yet");
    c.send(&remember_forgotten(6, [7; 32])).await;
    assert_eq!(c.recv().await.1, ControlPayload::ForgottenRemembered);
    c.send(&remember_forgotten(7, [7; 32])).await;
    assert_eq!(c.recv().await.1, ControlPayload::ForgottenRemembered, "a repeat is fine");
    c.send(&is_forgotten(8, Some([7; 32]))).await;
    let (_, _, bytes) = c.recv().await;
    assert!(read_envelope(&bytes).unwrap().payload_as_forgotten_status().unwrap().forgotten());
    c.send(&is_forgotten(9, Some([8; 32]))).await;
    let (_, _, bytes) = c.recv().await;
    assert!(!read_envelope(&bytes).unwrap().payload_as_forgotten_status().unwrap().forgotten(), "another name");
    c.send(&is_forgotten(10, None)).await;
    assert_eq!(c.recv_error().await.1, ErrorCode::BadRequest);
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn query_wire_budget_pages_large_headers_without_losing_rows_or_closing_the_connection() {
    let broker = Broker::start(Duration::from_secs(30)).await;
    let (mut p, cam) = producer(&broker).await;
    for n in 1..=3u8 {
        let mut spec = continuum_core::HeaderSpec::raw(
            continuum_core::Uuid(id_of(n)),
            continuum_core::StreamId(cam),
            u64::from(n) * 100,
            u64::from(n) * 100 + 10,
        );
        spec.shape_name = "x".repeat(6 * 1024 * 1024);
        let header = continuum_core::HeaderBuf::build(&spec).unwrap().into_bytes();
        let notice = envelope(u64::from(n), |b| {
            let h = b.create_vector(&header);
            let loc = chunk_at(continuum_core::StorageTier::Tier1Warm).build_into(b);
            let notice = IngestNotice::create(
                b,
                &IngestNoticeArgs { stream_id: cam, header_bytes: Some(h), locator: Some(loc), ..Default::default() },
            );
            (ControlPayload::IngestNotice, notice.as_union_value())
        });
        p.send(&notice).await;
        ingested(&mut p).await;
    }
    let mut c = consumer(&broker).await;
    let mut after = 0;
    let mut all = Vec::new();
    for request in 10..13 {
        c.send(&SeqQuery::after("cam0", after).build(request)).await;
        let (id, kind, bytes) = c.recv().await;
        assert_eq!((id, kind), (request, ControlPayload::QueryResult));
        assert!(bytes.len() <= continuum_wire::MAX_FRAME);
        let r = read_envelope(&bytes).unwrap().payload_as_query_result().unwrap();
        let rows = r.items().unwrap();
        for row in rows.iter() {
            let h = continuum_core::HeaderBuf::new(row.header_bytes().unwrap().bytes().to_vec()).unwrap();
            all.push(h.id().0[15]);
        }
        if r.next_after() == 0 {
            assert!(r.complete());
            break;
        }
        assert!(r.truncated());
        assert!(!r.complete());
        assert_eq!(r.next_after(), rows.get(rows.len() - 1).commit_seq());
        after = r.next_after();
    }
    assert_eq!(all, [1, 2, 3]);
    c.send(&query_for(20, "cam0", 0, 1_000, fbs::QueryMode::HitOnly)).await;
    let (_, kind, bytes) = c.recv().await;
    assert_eq!(kind, ControlPayload::QueryResult);
    let r = read_envelope(&bytes).unwrap().payload_as_query_result().unwrap();
    assert_eq!(r.next_after(), 0);
    assert!(r.truncated());
    assert!(!r.complete());
    c.send(&SeqQuery::after("cam0", after).build(21)).await;
    seq_result(&mut c, 21).await;
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_unpageable_catalog_item_returns_a_size_error_without_disconnect() {
    use continuum_catalog::Catalog as _;
    let (catalog, cam) = catalog_with_frames();
    let mut spec = continuum_core::HeaderSpec::raw(continuum_core::Uuid(id_of(9)), cam, 100, 110);
    spec.shape_name = "x".repeat(continuum_wire::MAX_FRAME);
    catalog
        .insert(vec![continuum_catalog::NewPayload {
            header: continuum_core::HeaderBuf::build(&spec).unwrap(),
            location: Some(chunk_at(continuum_core::StorageTier::Tier1Warm)),
            coverage: None,
        }])
        .unwrap();
    let broker = Broker::start_on(Duration::from_secs(30), 16, Box::new(catalog)).await;
    let mut c = consumer(&broker).await;
    let mut q = SeqQuery::after("cam0", 0);
    q.ids = &[9];
    c.send(&q.build(30)).await;
    let (id, kind, bytes) = c.recv().await;
    assert_eq!((id, kind), (30, ControlPayload::Error));
    let error = read_envelope(&bytes).unwrap().payload_as_error().unwrap();
    assert_eq!(error.code(), ErrorCode::BadRequest);
    assert!(error.message().unwrap().contains("size limit"));
    let mut q = SeqQuery::after("cam0", 0);
    q.ids = &[1];
    c.send(&q.build(31)).await;
    assert_eq!(seq_result(&mut c, 31).await.0.len(), 1);
    broker.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_single_item_near_the_frame_limit_uses_exact_size_instead_of_conservative_overhead() {
    use continuum_catalog::Catalog as _;
    let (catalog, cam) = catalog_with_frames();
    let mut spec = continuum_core::HeaderSpec::raw(continuum_core::Uuid(id_of(9)), cam, 100, 110);
    spec.shape_name = "x".repeat(continuum_wire::MAX_FRAME - 512);
    catalog
        .insert(vec![continuum_catalog::NewPayload {
            header: continuum_core::HeaderBuf::build(&spec).unwrap(),
            location: Some(chunk_at(continuum_core::StorageTier::Tier1Warm)),
            coverage: None,
        }])
        .unwrap();
    let broker = Broker::start_on(Duration::from_secs(30), 16, Box::new(catalog)).await;
    let mut c = consumer(&broker).await;
    let mut q = SeqQuery::after("cam0", 0);
    q.ids = &[9];
    c.send(&q.build(40)).await;
    assert_eq!(seq_result(&mut c, 40).await.0.len(), 1);
    broker.stop().await;
}

/// The dashboard's face snapshot asked for a whole window in one sequence page (`max_items` 40,000) of many
/// small inline rows. When only inline bytes were budgeted, headers pushed the page to about 18 MB and the
/// session closed on every retry. The page is cut under the frame limit and the drain still returns every row.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_large_max_items_page_of_small_inline_rows_is_cut_under_the_frame_limit() {
    use continuum_catalog::Catalog as _;
    const ROWS: u16 = 4000;
    let (catalog, cam) = catalog_with_frames();
    let rows: Vec<_> = (1..=ROWS)
        .map(|n| {
            let mut id = [0u8; 16];
            id[6] = 0x70;
            id[13] = 0xAB;
            id[14..16].copy_from_slice(&n.to_be_bytes());
            let t = 1_000 + u64::from(n) * 10;
            let mut spec = continuum_core::HeaderSpec::raw(continuum_core::Uuid(id), cam, t, t + 5);
            spec.shape_name = "f".repeat(4096);
            continuum_catalog::NewPayload {
                header: continuum_core::HeaderBuf::build(&spec).unwrap(),
                location: Some(continuum_core::StorageLocator::Inline { bytes: vec![b'x'; 1024] }),
                coverage: None,
            }
        })
        .collect();
    catalog.insert(rows).unwrap();
    let broker = Broker::start_on(Duration::from_secs(30), 16, Box::new(catalog)).await;
    let mut c = consumer(&broker).await;
    let (mut after, mut seen, mut pages) = (0, Vec::new(), 0);
    loop {
        let mut q = SeqQuery::after("cam0", after);
        q.max_items = 40_000;
        c.send(&q.build(100 + pages)).await;
        let (id, kind, bytes) = c.recv().await;
        assert_eq!((id, kind), (100 + pages, ControlPayload::QueryResult));
        assert!(bytes.len() <= continuum_wire::MAX_FRAME, "page of {} bytes", bytes.len());
        pages += 1;
        let r = read_envelope(&bytes).unwrap().payload_as_query_result().unwrap();
        for row in r.items().unwrap().iter() {
            let h = continuum_core::HeaderBuf::new(row.header_bytes().unwrap().bytes().to_vec()).unwrap();
            if h.id().0[13] == 0xAB {
                seen.push(u16::from_be_bytes([h.id().0[14], h.id().0[15]]));
            }
        }
        if r.next_after() == 0 {
            break;
        }
        assert!(r.truncated());
        after = r.next_after();
    }
    assert!(pages > 1, "the rows were built to overflow one frame");
    assert_eq!(seen, (1..=ROWS).collect::<Vec<_>>());
    broker.stop().await;
}
