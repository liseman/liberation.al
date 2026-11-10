//! A stand-in broker for the client tests: it accepts one connection, reads `Hello`, and sends a scripted reply
//! (with fds), for what the real broker doesn't send yet.
#![allow(dead_code)] // each test binary uses some of these

use std::future::Future;
use std::os::fd::{AsFd, OwnedFd};
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};

use continuum_core::schema::continuum as fb;
use continuum_core::{SCHEMA_MAJOR, SCHEMA_MINOR};
use continuum_shm::{Arena, ArenaConfig, SlabClass};
use continuum_wire::{read_envelope, read_frame, write_frame_with_fds};
use flatbuffers::FlatBufferBuilder;
use tokio::net::{UnixListener, UnixStream};

pub fn arena_name() -> String {
    static N: AtomicU32 = AtomicU32::new(0);
    format!("/cnt.c.{}.{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed))
}

pub struct ShmArena(pub Arena);

impl Drop for ShmArena {
    fn drop(&mut self) {
        Arena::unlink(self.0.name()).ok();
    }
}

pub fn make_arena(generation: u32) -> ShmArena {
    let config = ArenaConfig {
        classes: vec![SlabClass { slot_size: 64 * 1024, slot_count: 2 }],
        generation,
        ..ArenaConfig::default()
    };
    ShmArena(Arena::create(&arena_name(), &config).unwrap())
}

/// A `Welcome` describing `arenas` as (name, generation, size).
pub fn welcome(arenas: &[(&str, u32, u64)]) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let infos: Vec<_> = arenas
        .iter()
        .map(|&(name, generation, size_bytes)| {
            let name = b.create_string(name);
            let class = fb::SizeClass::create(&mut b, &fb::SizeClassArgs { slab_bytes: 64 * 1024, count: 2 });
            let classes = b.create_vector(&[class]);
            fb::ArenaInfo::create(
                &mut b,
                &fb::ArenaInfoArgs { name: Some(name), generation, size_bytes, classes: Some(classes) },
            )
        })
        .collect();
    let infos = b.create_vector(&infos);
    let schema = fb::SchemaVersion::create(&mut b, &fb::SchemaVersionArgs { major: SCHEMA_MAJOR, minor: SCHEMA_MINOR });
    let id = fb::Uuid::new(&[7; 16]);
    let w = fb::Welcome::create(
        &mut b,
        &fb::WelcomeArgs { session_id: Some(&id), schema: Some(schema), arenas: Some(infos), ..Default::default() },
    );
    let env = fb::ControlEnvelope::create(
        &mut b,
        &fb::ControlEnvelopeArgs {
            request_id: 0,
            payload_type: fb::ControlPayload::Welcome,
            payload: Some(w.as_union_value()),
        },
    );
    b.finish(env, None);
    b.finished_data().to_vec()
}

pub fn error_reply(code: fb::ErrorCode, message: &str) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let message = b.create_string(message);
    let e = fb::Error::create(&mut b, &fb::ErrorArgs { code, message: Some(message) });
    let env = fb::ControlEnvelope::create(
        &mut b,
        &fb::ControlEnvelopeArgs {
            request_id: 0,
            payload_type: fb::ControlPayload::Error,
            payload: Some(e.as_union_value()),
        },
    );
    b.finish(env, None);
    b.finished_data().to_vec()
}

/// Accepts one connection on `socket`, reads its Hello, and answers with `reply` carrying `fds`.
pub async fn stand_in(socket: &Path, reply: Vec<u8>, fds: Vec<std::os::fd::OwnedFd>) -> tokio::task::JoinHandle<()> {
    let listener = UnixListener::bind(socket).unwrap();
    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut buf = Vec::new();
        read_frame(&mut stream, &mut buf).await.unwrap();
        let borrowed: Vec<_> = fds.iter().map(AsFd::as_fd).collect();
        write_frame_with_fds(&mut stream, &reply, &borrowed).await.unwrap();
        // Hold the connection open until the client goes.
        let _ = read_frame(&mut stream, &mut buf).await;
    })
}

/// A fresh fd for `a`'s shared memory object, as the broker would hold one.
pub fn dup(a: &Arena) -> std::os::fd::OwnedFd {
    let cname = std::ffi::CString::new(a.name()).unwrap();
    // SAFETY: a valid NUL-terminated name.
    let raw = unsafe { libc::shm_open(cname.as_ptr(), libc::O_RDWR, 0) };
    assert!(raw >= 0);
    // SAFETY: a fresh fd we own.
    unsafe { std::os::fd::FromRawFd::from_raw_fd(raw) }
}

/// Like [`stand_in`], but after the reply the stream goes to `script`.
pub async fn scripted<F, Fut>(
    socket: &Path,
    reply: Vec<u8>,
    fds: Vec<OwnedFd>,
    script: F,
) -> tokio::task::JoinHandle<()>
where
    F: FnOnce(UnixStream) -> Fut + Send + 'static,
    Fut: Future<Output = ()> + Send,
{
    let listener = UnixListener::bind(socket).unwrap();
    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut buf = Vec::new();
        read_frame(&mut stream, &mut buf).await.unwrap();
        let borrowed: Vec<_> = fds.iter().map(AsFd::as_fd).collect();
        write_frame_with_fds(&mut stream, &reply, &borrowed).await.unwrap();
        drop(fds);
        script(stream).await;
    })
}

/// Reads one request: its id, payload type, and (for a Heartbeat) its telemetry's queue depth. None at the close.
pub async fn read_request(stream: &mut UnixStream) -> Option<(u64, fb::ControlPayload, u32, String)> {
    let mut buf = Vec::new();
    read_frame(stream, &mut buf).await.ok()?;
    let env = read_envelope(&buf).unwrap();
    let depth = env.payload_as_heartbeat().and_then(|h| h.telemetry()).map_or(0, |t| t.queue_depth());
    let reason = env.payload_as_goodbye().and_then(|g| g.reason()).unwrap_or_default().to_owned();
    Some((env.request_id(), env.payload_type(), depth, reason))
}

/// A `Heartbeat` envelope with `request_id` and the clock value `now_ns`.
pub fn heartbeat_reply(request_id: u64, now_ns: u64) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let h = fb::Heartbeat::create(&mut b, &fb::HeartbeatArgs { now_ns, telemetry: None });
    let env = fb::ControlEnvelope::create(
        &mut b,
        &fb::ControlEnvelopeArgs {
            request_id,
            payload_type: fb::ControlPayload::Heartbeat,
            payload: Some(h.as_union_value()),
        },
    );
    b.finish(env, None);
    b.finished_data().to_vec()
}

/// A `Welcome` with no schema version.
pub fn welcome_without_schema() -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let id = fb::Uuid::new(&[7; 16]);
    let w = fb::Welcome::create(
        &mut b,
        &fb::WelcomeArgs { session_id: Some(&id), schema: None, arenas: None, ..Default::default() },
    );
    let env = fb::ControlEnvelope::create(
        &mut b,
        &fb::ControlEnvelopeArgs {
            request_id: 0,
            payload_type: fb::ControlPayload::Welcome,
            payload: Some(w.as_union_value()),
        },
    );
    b.finish(env, None);
    b.finished_data().to_vec()
}

/// The number of open fds in this process (/dev/fd, on Linux and macOS).
pub fn open_fds() -> usize {
    std::fs::read_dir("/dev/fd").unwrap().count()
}
