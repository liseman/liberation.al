//! File descriptors over many sessions, alone in its own test binary: counting `/proc/self/fd` while other tests
//! open and close sockets in the same process would fail with no leak, or miss a real one.
#![cfg(target_os = "linux")]

use std::time::Duration;

use continuum_broker::protocol::{SCHEMA_MAJOR, read_envelope};
use continuum_broker::server::{Config, Server};
use continuum_core::schema::continuum::{
    ControlEnvelope, ControlEnvelopeArgs, ControlPayload, Goodbye, GoodbyeArgs, Hello, HelloArgs, SchemaVersion,
    SchemaVersionArgs,
};
use continuum_wire::{read_frame, read_frame_with_fds, write_frame};
use flatbuffers::FlatBufferBuilder;
use tokio::net::UnixStream;

fn open_fds() -> usize {
    std::fs::read_dir("/proc/self/fd").unwrap().count()
}

fn frame(payload_type: ControlPayload, hello: bool) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let payload = if hello {
        let schema = SchemaVersion::create(&mut b, &SchemaVersionArgs { major: SCHEMA_MAJOR, minor: 0 });
        Hello::create(&mut b, &HelloArgs { schema: Some(schema), pid: std::process::id(), ..Default::default() })
            .as_union_value()
    } else {
        Goodbye::create(&mut b, &GoodbyeArgs { reason: None }).as_union_value()
    };
    let env =
        ControlEnvelope::create(&mut b, &ControlEnvelopeArgs { request_id: 1, payload_type, payload: Some(payload) });
    b.finish(env, None);
    b.finished_data().to_vec()
}

/// One session: Hello, Welcome (taking the arena's fd and mapping it, or reading without room for fds), Goodbye.
async fn session(socket: &std::path::Path, map: bool) {
    let mut s = UnixStream::connect(socket).await.unwrap();
    write_frame(&mut s, &frame(ControlPayload::Hello, true)).await.unwrap();
    let mut buf = Vec::new();
    if map {
        use std::os::fd::AsFd;
        let fds = read_frame_with_fds(&mut s, &mut buf, 4).await.unwrap();
        assert_eq!(fds.len(), 1);
        let env = read_envelope(&buf).unwrap();
        let name = env.payload_as_welcome().unwrap().arenas().unwrap().get(0).name().unwrap().to_owned();
        drop(continuum_shm::Arena::from_fd(fds[0].as_fd(), &name).unwrap());
    } else {
        // No room for fds: the kernel discards them.
        read_frame(&mut s, &mut buf).await.unwrap();
    }
    assert_eq!(read_envelope(&buf).unwrap().payload_type(), ControlPayload::Welcome);
    write_frame(&mut s, &frame(ControlPayload::Goodbye, false)).await.unwrap();
    // The broker closes the connection.
    assert!(matches!(tokio::time::timeout(Duration::from_secs(5), read_frame(&mut s, &mut buf)).await, Ok(Err(_))));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sessions_coming_and_going_leak_no_fds() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("broker.sock");
    let mut config = Config::new(socket.clone(), Duration::from_secs(30));
    config.tier0 = Some(vec![continuum_sched::SlotClass { size_bytes: 16_384, count: 4 }]);
    let server = Server::bind(config).await.unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(server.run_until(async {
        let _ = stopped.await;
    }));
    // Warm up: the first session may open things that stay (runtime threads, timers).
    session(&socket, true).await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    let before = open_fds();
    for i in 0..40 {
        session(&socket, i % 2 == 0).await;
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    let after = open_fds();
    assert_eq!(after, before, "{before} fds before 40 sessions, {after} after");
    let _ = stop.send(());
    task.await.unwrap();
}
