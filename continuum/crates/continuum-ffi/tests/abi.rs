//! The C ABI called the way C calls it: status codes, error messages, ownership, leases, slots, queries, promises,
//! subscriptions, and jobs, against a real broker (and a stand-in broker for an arena of a chosen shape).

use std::ffi::{CStr, CString};
use std::os::fd::{AsFd, FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use continuum::*;
use continuum_broker::server::{Config, Server};
use continuum_core::schema::continuum as fb;
use continuum_core::{SCHEMA_MAJOR, SCHEMA_MINOR};
use continuum_shm::{Arena, ArenaConfig, SlabClass};
use flatbuffers::FlatBufferBuilder;

/// A real broker on its own thread and runtime (the ABI's calls block, so they can't run inside a test's runtime).
struct Broker {
    _dir: tempfile::TempDir,
    socket: CString,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Broker {
    fn drop(&mut self) {
        let _ = self.stop.take().map(|s| s.send(()));
        let _ = self.thread.take().map(std::thread::JoinHandle::join);
    }
}

fn broker() -> Broker {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("b.sock");
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let (ready, is_ready) = std::sync::mpsc::channel();
    let p = path.clone();
    let thread = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(1).enable_all().build().unwrap();
        rt.block_on(async move {
            let server = Server::bind(Config::new(p, Duration::from_secs(30))).await.unwrap();
            ready.send(()).unwrap();
            server
                .run_until(async {
                    let _ = stopped.await;
                })
                .await;
        });
    });
    is_ready.recv().unwrap();
    Broker { _dir: dir, socket: c(&path), stop: Some(stop), thread: Some(thread) }
}

fn c(path: &Path) -> CString {
    CString::new(path.to_str().unwrap()).unwrap()
}

fn last_error() -> String {
    // SAFETY: the library returns a valid NUL-terminated string.
    unsafe { CStr::from_ptr(cnt_last_error_message()) }.to_string_lossy().into_owned()
}

fn connect(socket: &CString) -> *mut cnt_client {
    let mut client = ptr::null_mut();
    // SAFETY: valid arguments.
    let s = unsafe { cnt_connect(socket.as_ptr(), ptr::null(), &raw mut client) };
    assert_eq!(s, cnt_status::CNT_OK, "{}", last_error());
    client
}

fn spec(name: &CString, shape: &CString) -> cnt_stream_spec {
    cnt_stream_spec {
        name: name.as_ptr(),
        kind: CNT_DERIVED,
        format: 0,
        layer: 1,
        shape_name: shape.as_ptr(),
        shape_version: 1,
        nominal_rate_hz: 0.0,
        storage: CNT_SLAB,
        slab_class: 0,
        ring_stride_bytes: 0,
        ring_seconds: 0.0,
        has_anchor: 0,
        anchor_boot_id: [0; 16],
        anchor_monotonic_ns: 0,
        anchor_utc_ns: 0,
    }
}

fn register(client: *mut cnt_client, spec: &cnt_stream_spec) -> Result<cnt_stream_registered, cnt_status> {
    // SAFETY: an all-zero struct is a valid value to overwrite.
    let mut out: cnt_stream_registered = unsafe { std::mem::zeroed() };
    // SAFETY: valid arguments.
    match unsafe { cnt_register_stream(client, spec, &raw mut out) } {
        cnt_status::CNT_OK => Ok(out),
        s => Err(s),
    }
}

#[test]
fn the_version_functions_agree_with_the_header() {
    assert_eq!(cnt_abi_version(), CNT_ABI_VERSION);
    // SAFETY: a static NUL-terminated string.
    assert_eq!(unsafe { CStr::from_ptr(cnt_version()) }.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
}

#[test]
fn a_session_heartbeats_and_registers_streams() {
    let b = broker();
    let client = connect(&b.socket);
    let mut id = [0u8; 16];
    // SAFETY: valid arguments.
    assert_eq!(unsafe { cnt_session_id(client, id.as_mut_ptr()) }, cnt_status::CNT_OK);
    assert_eq!(id[6] >> 4, 7, "a UUIDv7");
    let mut now = 0;
    // SAFETY: valid arguments.
    assert_eq!(unsafe { cnt_heartbeat(client, &raw mut now) }, cnt_status::CNT_OK);
    assert!(now > 1_700_000_000_000_000_000);
    assert_eq!(last_error(), "", "a call that worked leaves no message");

    let (name, shape, other) =
        (CString::new("pose.body").unwrap(), CString::new("pose").unwrap(), CString::new("x").unwrap());
    let first = register(client, &spec(&name, &shape)).unwrap();
    let again = register(client, &spec(&name, &shape)).unwrap();
    assert_eq!(first.stream_id, again.stream_id);
    assert_eq!(register(client, &spec(&name, &other)).unwrap_err(), cnt_status::CNT_BROKER);
    assert_eq!(cnt_last_broker_code(), fb::ErrorCode::Conflict.0, "{}", last_error());
    assert!(last_error().contains("Conflict"));

    // SAFETY: a live client, then freed once.
    unsafe {
        assert_eq!(cnt_client_close(client, ptr::null()), cnt_status::CNT_OK);
        assert_eq!(cnt_heartbeat(client, ptr::null_mut()), cnt_status::CNT_CLOSED, "requests after close fail");
        cnt_client_free(client);
    }
}

#[test]
fn bad_arguments_are_refused_with_a_message() {
    let mut client = ptr::dangling_mut::<cnt_client>();
    // SAFETY: a null socket is a documented error, and out is writable.
    assert_eq!(unsafe { cnt_connect(ptr::null(), ptr::null(), &raw mut client) }, cnt_status::CNT_INVALID_ARGUMENT);
    assert!(client.is_null(), "out is cleared on failure");
    assert!(last_error().contains("socket"));
    let b = broker();
    let tiny = cnt_connect_options { heartbeat_interval_ms: 1, ..cnt_connect_options_default() };
    // SAFETY: valid pointers.
    let too_fast = unsafe { cnt_connect(b.socket.as_ptr(), &raw const tiny, &raw mut client) };
    assert_eq!(too_fast, cnt_status::CNT_INVALID_ARGUMENT);
    // SAFETY: a null out is a documented error.
    let no_out = unsafe { cnt_connect(b.socket.as_ptr(), ptr::null(), ptr::null_mut()) };
    assert_eq!(no_out, cnt_status::CNT_INVALID_ARGUMENT);

    let client = connect(&b.socket);
    let (name, shape) = (CString::new("s").unwrap(), CString::new("t").unwrap());
    for bad in [
        cnt_stream_spec { kind: 7, ..spec(&name, &shape) },
        cnt_stream_spec { storage: 9, ..spec(&name, &shape) },
        cnt_stream_spec { format: 200, ..spec(&name, &shape) },
        cnt_stream_spec { layer: 9, ..spec(&name, &shape) },
        cnt_stream_spec { name: ptr::null(), ..spec(&name, &shape) },
    ] {
        assert_eq!(register(client, &bad).unwrap_err(), cnt_status::CNT_INVALID_ARGUMENT, "{}", last_error());
    }
    // SAFETY: null handles are documented no-ops or errors; the client is freed once.
    unsafe {
        assert_eq!(cnt_heartbeat(ptr::null_mut(), ptr::null_mut()), cnt_status::CNT_INVALID_ARGUMENT);
        cnt_client_free(ptr::null_mut());
        assert_eq!(cnt_lease_release(ptr::null_mut()), cnt_status::CNT_OK);
        cnt_client_free(client);
    }
}

#[test]
fn connecting_to_nothing_is_an_io_error() {
    let dir = tempfile::tempdir().unwrap();
    let mut client = ptr::null_mut();
    let socket = c(&dir.path().join("none.sock"));
    // SAFETY: valid arguments.
    assert_eq!(unsafe { cnt_connect(socket.as_ptr(), ptr::null(), &raw mut client) }, cnt_status::CNT_IO);
    assert!(client.is_null());
}

// ---- Leases, through a stand-in broker that hands out one arena ----

fn arena_name() -> String {
    static N: AtomicU32 = AtomicU32::new(0);
    format!("/cnt.f.{}.{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed))
}

fn welcome(name: &str, generation: u32, size: u64) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let n = b.create_string(name);
    let info = fb::ArenaInfo::create(
        &mut b,
        &fb::ArenaInfoArgs { name: Some(n), generation, size_bytes: size, classes: None },
    );
    let arenas = b.create_vector(&[info]);
    let schema = fb::SchemaVersion::create(&mut b, &fb::SchemaVersionArgs { major: SCHEMA_MAJOR, minor: SCHEMA_MINOR });
    let id = fb::Uuid::new(&[7; 16]);
    let w = fb::Welcome::create(
        &mut b,
        &fb::WelcomeArgs { session_id: Some(&id), schema: Some(schema), arenas: Some(arenas), ..Default::default() },
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

/// Serves one connection: reads Hello, sends Welcome with the arena's fd, then reads until the client goes.
fn stand_in(socket: PathBuf, reply: Vec<u8>, fd: OwnedFd) -> std::thread::JoinHandle<()> {
    let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async move {
            listener.set_nonblocking(true).unwrap();
            let listener = tokio::net::UnixListener::from_std(listener).unwrap();
            let (mut s, _) = listener.accept().await.unwrap();
            let mut buf = Vec::new();
            continuum_wire::read_frame(&mut s, &mut buf).await.unwrap();
            continuum_wire::write_frame_with_fds(&mut s, &reply, &[fd.as_fd()]).await.unwrap();
            while continuum_wire::read_frame(&mut s, &mut buf).await.is_ok() {}
        });
    })
}

#[test]
fn a_lease_pins_a_slot_reads_it_in_place_and_outlives_its_client() {
    let name = arena_name();
    let config = ArenaConfig {
        classes: vec![SlabClass { slot_size: 64 * 1024, slot_count: 2 }],
        generation: 3,
        ..Default::default()
    };
    let arena = Arena::create(&name, &config).unwrap();
    let mut slot = arena.claim(0, 1).unwrap();
    slot.as_mut_slice()[..5].copy_from_slice(b"frame");
    let handle = slot.publish(5).unwrap();
    let cname = CString::new(name.as_str()).unwrap();
    // SAFETY: a valid name; the fd is owned by OwnedFd.
    let fd = unsafe { OwnedFd::from_raw_fd(libc::shm_open(cname.as_ptr(), libc::O_RDWR, 0)) };
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    let server = stand_in(socket.clone(), welcome("hot", 3, arena.len() as u64), fd);

    let client = connect(&c(&socket));
    // SAFETY: a live client throughout; every handle is freed once.
    unsafe {
        assert_eq!(cnt_arena_count(client), 1);
        // SAFETY: an all-zero struct is a valid value to overwrite.
        let mut info: cnt_arena_info = std::mem::zeroed();
        assert_eq!(cnt_arena_info_at(client, 0, &raw mut info), cnt_status::CNT_OK);
        assert_eq!(CStr::from_ptr(info.name.as_ptr()).to_str().unwrap(), "hot");
        assert_eq!((info.generation, info.size_bytes), (3, arena.len() as u64));
        assert_eq!(info.base as usize % 16384, 0);
        assert_eq!(cnt_arena_info_at(client, 1, &raw mut info), cnt_status::CNT_NOT_FOUND);

        let r = cnt_slot_ref { slab_class: handle.class, index: handle.index, generation: handle.generation };
        let mut lease: *mut cnt_lease = ptr::null_mut();
        assert_eq!(cnt_lease_slot(client, 0, r, 5_000, &raw mut lease), cnt_status::CNT_OK, "{}", last_error());
        let data = std::slice::from_raw_parts(cnt_lease_data(lease), cnt_lease_len(lease));
        assert_eq!(data, b"frame");
        assert_eq!(cnt_lease_data(lease) as usize % 16384, 0);
        assert_eq!(arena.pin_count(handle).unwrap(), 1, "the pin shows in the broker's mapping");

        let stale = cnt_slot_ref { generation: handle.generation + 1, ..r };
        let mut none: *mut cnt_lease = ptr::dangling_mut();
        assert_eq!(cnt_lease_slot(client, 0, stale, 5_000, &raw mut none), cnt_status::CNT_STALE);
        assert!(none.is_null());
        assert_eq!(cnt_lease_slot(client, 5, r, 5_000, &raw mut none), cnt_status::CNT_NOT_FOUND);

        cnt_client_free(client); // the lease keeps the arena mapped
        assert!(cnt_lease_valid(lease));
        assert_eq!(std::slice::from_raw_parts(cnt_lease_data(lease), 5), b"frame");
        assert_eq!(cnt_lease_release(lease), cnt_status::CNT_OK);
    }
    assert_eq!(arena.pin_count(handle).unwrap(), 0, "released");
    server.join().unwrap();
    Arena::unlink(&name).ok();
}

#[test]
fn errors_are_per_thread() {
    let mut client = ptr::null_mut();
    // SAFETY: a null socket is a documented error.
    unsafe { cnt_connect(ptr::null(), ptr::null(), &raw mut client) };
    assert!(!last_error().is_empty());
    let other = std::thread::spawn(last_error).join().unwrap();
    assert_eq!(other, "", "another thread's last error is its own");
}

// ---- The worker side ----

fn capability(produces: &CString, domain: u8) -> cnt_capability {
    cnt_capability {
        produces: produces.as_ptr(),
        inputs: ptr::null(),
        n_inputs: 0,
        compute_domain: domain,
        ram_mb: 0,
        vram_mb: 0,
        est_ms_per_item: 0.0,
        cold_start_ms: 0,
        max_batch: 1,
        preemptible: 1,
        role: 0,
        output_bytes: 0,
        outputs_per_job: 1,
        has_expected_confidence: 0,
        expected_confidence: 0.0,
    }
}

/// A client with a "pose" stream and a worker that produces it; returns the client, pose's id, and the capability id.
fn worker(socket: &CString, domain: u8) -> (*mut cnt_client, u32, u32) {
    let client = connect(socket);
    let (pose, shape) = (CString::new("pose").unwrap(), CString::new("pose").unwrap());
    let pose_id = register(client, &spec(&pose, &shape)).unwrap().stream_id;
    let caps = [capability(&pose, domain)];
    let mut results = [cnt_capability_result::default()];
    let id = CString::new("worker:pose").unwrap();
    // SAFETY: valid arguments.
    let s = unsafe {
        cnt_register_worker(
            client,
            id.as_ptr(),
            ptr::null(),
            -1,
            0,
            caps.as_ptr(),
            1,
            results.as_mut_ptr(),
            ptr::null_mut(),
            0,
        )
    };
    assert_eq!(s, cnt_status::CNT_OK, "{}", last_error());
    assert_eq!(results[0].accepted, 1);
    (client, pose_id, results[0].id)
}

fn quote(client: *mut cnt_client, cap: u32, band: u8, domain: u8) -> *mut cnt_work_lease {
    let stage = CString::new("pose").unwrap();
    let spec = cnt_quote_spec {
        capability_id: cap,
        stage: stage.as_ptr(),
        band,
        compute_domain: domain,
        output_bytes: 0,
        est_duration_ms: 0,
        pin_parents: ptr::null(),
        n_pin_parents: 0,
        range_start_ns: 0,
        range_end_ns: 1_000,
    };
    let mut lease = ptr::null_mut();
    // SAFETY: valid arguments.
    let s = unsafe { cnt_quote(client, &raw const spec, &raw mut lease, ptr::null_mut()) };
    assert_eq!(s, cnt_status::CNT_OK, "{}", last_error());
    assert!(!lease.is_null());
    lease
}

fn output_header(stream: u32, n: u64) -> continuum_core::HeaderBuf {
    let mut id = [0u8; 16];
    id[6] = 0x70;
    id[8..].copy_from_slice(&(0x8000_0000_0000_0000 | n).to_be_bytes());
    continuum_core::HeaderBuf::build(&continuum_core::HeaderSpec::raw(
        continuum_core::Uuid(id),
        continuum_core::StreamId(stream),
        100,
        200,
    ))
    .unwrap()
}

#[test]
fn a_worker_registers_quotes_renews_commits_and_aborts() {
    let b = broker();
    let (client, pose, cap) = worker(&b.socket, 0);

    let lease = quote(client, cap, 2, 0);
    // SAFETY: a live lease.
    let first = unsafe { cnt_work_lease_expires_at_ns(lease) };
    assert!(first > 0);
    std::thread::sleep(Duration::from_millis(20));
    let mut renewed = 0;
    // SAFETY: valid arguments.
    let s = unsafe { cnt_work_lease_renew(lease, 30_000, &raw mut renewed) };
    assert_eq!(s, cnt_status::CNT_OK, "{}", last_error());
    assert!(renewed > first);
    // SAFETY: a live lease, on this thread only.
    assert_eq!(unsafe { cnt_work_lease_keep_alive(lease, 50, 30_000) }, cnt_status::CNT_OK);
    std::thread::sleep(Duration::from_millis(300));
    // SAFETY: a live lease.
    assert!(unsafe { cnt_work_lease_expires_at_ns(lease) } > renewed, "the keep-alive renewed it");

    // A bad argument sends nothing and leaves the lease the caller's.
    let junk = [cnt_output { header: b"not a header".as_ptr(), header_len: 12, at: ptr::null() }];
    // SAFETY: valid arguments (the header is junk on purpose); the lease stays live on a bad argument.
    let s = unsafe { cnt_work_lease_commit(lease, junk.as_ptr(), 1, ptr::null(), ptr::null()) };
    assert_eq!(s, cnt_status::CNT_INVALID_ARGUMENT, "{}", last_error());
    // SAFETY: still a live lease.
    assert!(unsafe { cnt_work_lease_expires_at_ns(lease) } > 0, "still held");

    let header = output_header(pose, 1);
    let bytes = b"a pose";
    let at = cnt_location {
        kind: CNT_LOCATION_INLINE,
        inline_bytes: bytes.as_ptr(),
        inline_len: bytes.len(),
        arena: [0; CNT_NAME_MAX],
        arena_generation: 0,
        slab_class: 0,
        slot: 0,
        slot_generation: 0,
        offset: 0,
        length: 0,
    };
    let outputs =
        [cnt_output { header: header.as_bytes().as_ptr(), header_len: header.as_bytes().len(), at: &raw const at }];
    let device = CString::new("CPU").unwrap();
    let cost = cnt_cost {
        wall_time_us: 900,
        queue_wait_us: 0,
        ram_bytes: 0,
        vram_bytes: 0,
        device: device.as_ptr(),
        items: 1,
    };
    // SAFETY: valid arguments; the lease isn't used again.
    let s = unsafe { cnt_work_lease_commit(lease, outputs.as_ptr(), 1, ptr::null(), &raw const cost) };
    assert_eq!(s, cnt_status::CNT_OK, "{}", last_error());

    let lease = quote(client, cap, 2, 0);
    let reason = CString::new("out of memory").unwrap();
    // SAFETY: valid arguments; the lease isn't used again.
    assert_eq!(unsafe { cnt_work_lease_abort(lease, reason.as_ptr()) }, cnt_status::CNT_OK, "{}", last_error());

    // Released without committing, and a null release is a no-op.
    // SAFETY: a live lease, not used again; then null.
    unsafe {
        cnt_work_lease_release(quote(client, cap, 2, 0));
        cnt_work_lease_release(ptr::null_mut());
        assert_eq!(cnt_client_close(client, ptr::null()), cnt_status::CNT_OK);
        cnt_client_free(client);
    }
}

#[test]
fn a_voice_turn_preempts_a_gpu_lease_and_both_events_arrive() {
    let b = broker();
    let (client, _, cap) = worker(&b.socket, 1);
    let lease = quote(client, cap, 3, 1);
    let voice = connect(&b.socket);
    // SAFETY: a live client.
    assert_eq!(unsafe { cnt_set_system_state(voice, 1) }, cnt_status::CNT_OK, "{}", last_error());

    // SAFETY: an all-zero event is a valid value to overwrite.
    let mut e: cnt_event = unsafe { std::mem::zeroed() };
    // SAFETY: valid arguments.
    assert_eq!(unsafe { cnt_next_event(client, 5_000, &raw mut e) }, cnt_status::CNT_OK, "{}", last_error());
    assert_eq!(e.r#type, CNT_EVENT_LEASE_PREEMPTED);
    // SAFETY: a live lease.
    assert_eq!(e.lease_id, unsafe { cnt_work_lease_id(lease) });
    assert_eq!((e.by_band, e.revoked), (0, 0));
    // SAFETY: the reason is NUL-terminated.
    assert!(!unsafe { CStr::from_ptr(e.reason.as_ptr()) }.is_empty());
    // SAFETY: valid arguments.
    assert_eq!(unsafe { cnt_next_event(client, 5_000, &raw mut e) }, cnt_status::CNT_OK, "{}", last_error());
    assert_eq!((e.r#type, e.voice_turn), (CNT_EVENT_SYSTEM_STATE_CHANGED, 1));
    assert!(e.changed_at_ns > 0);
    // SAFETY: valid arguments.
    assert_eq!(unsafe { cnt_next_event(client, 50, &raw mut e) }, cnt_status::CNT_TIMEOUT);

    // SAFETY: live handles, not used again.
    unsafe {
        cnt_work_lease_release(lease);
        cnt_client_free(voice);
        cnt_client_free(client);
    }
}

#[test]
fn bad_worker_arguments_are_refused_and_refusals_carry_reasons() {
    let b = broker();
    let client = connect(&b.socket);
    let (nothing, id) = (CString::new("nothing-here").unwrap(), CString::new("worker:x").unwrap());
    let caps = [capability(&nothing, 0)];
    let mut results = [cnt_capability_result { accepted: 9, id: 9, runnable: 9 }];
    let mut reasons = [0 as std::ffi::c_char; 256];
    // SAFETY: valid arguments.
    let s = unsafe {
        cnt_register_worker(
            client,
            id.as_ptr(),
            ptr::null(),
            -1,
            0,
            caps.as_ptr(),
            1,
            results.as_mut_ptr(),
            reasons.as_mut_ptr(),
            256,
        )
    };
    assert_eq!(s, cnt_status::CNT_OK, "{}", last_error());
    assert_eq!(results[0].accepted, 0);
    // SAFETY: NUL-terminated by the call.
    assert!(unsafe { CStr::from_ptr(reasons.as_ptr()) }.to_string_lossy().contains("nothing-here"));

    let bad = [cnt_capability { compute_domain: 9, ..capability(&nothing, 0) }];
    // SAFETY: valid arguments.
    let s = unsafe {
        cnt_register_worker(
            client,
            id.as_ptr(),
            ptr::null(),
            -1,
            0,
            bad.as_ptr(),
            1,
            results.as_mut_ptr(),
            ptr::null_mut(),
            0,
        )
    };
    assert_eq!(s, cnt_status::CNT_INVALID_ARGUMENT);
    assert!(last_error().contains("compute_domain"), "{}", last_error());
    // SAFETY: a live client.
    assert_eq!(unsafe { cnt_set_system_state(client, 9) }, cnt_status::CNT_INVALID_ARGUMENT);
    // SAFETY: a null lease, which the call refuses.
    let s = unsafe { cnt_work_lease_commit(ptr::null_mut(), ptr::null(), 0, ptr::null(), ptr::null()) };
    assert_eq!(s, cnt_status::CNT_INVALID_ARGUMENT);

    let empty = CString::new("").unwrap();
    let a = CString::new("a").unwrap();
    // SAFETY: valid strings, and null.
    unsafe {
        assert_eq!(cnt_stage_id(empty.as_ptr()), 0x811C_9DC5);
        assert_eq!(cnt_stage_id(a.as_ptr()), 0xE40C_292C);
        assert_eq!(cnt_stage_id(ptr::null()), 0);
        cnt_client_free(client);
    }
}

#[test]
fn a_work_lease_outliving_its_client_answers_closed_and_frees() {
    let b = broker();
    let (client, _, cap) = worker(&b.socket, 0);
    let lease = quote(client, cap, 2, 0);
    // SAFETY: a live client, not used again.
    unsafe { cnt_client_free(client) };
    let mut expires = 0;
    // SAFETY: the lease is still a live handle; its client is gone.
    assert_eq!(unsafe { cnt_work_lease_renew(lease, 1_000, &raw mut expires) }, cnt_status::CNT_CLOSED);
    // SAFETY: a live handle, not used again.
    unsafe { cnt_work_lease_release(lease) };
}

#[test]
fn out_of_range_counts_and_indexes_are_refused() {
    let b = broker();
    let (client, _, cap) = worker(&b.socket, 0);
    let stage = CString::new("pose").unwrap();
    let parents = [0u8; 16];
    let spec = cnt_quote_spec {
        capability_id: cap,
        stage: stage.as_ptr(),
        band: 2,
        compute_domain: 0,
        output_bytes: 0,
        est_duration_ms: 0,
        pin_parents: parents.as_ptr(),
        n_pin_parents: usize::MAX / 8,
        range_start_ns: 0,
        range_end_ns: 1_000,
    };
    let mut lease = ptr::null_mut();
    // SAFETY: valid pointers; the count is too large on purpose and is refused before it's used.
    let s = unsafe { cnt_quote(client, &raw const spec, &raw mut lease, ptr::null_mut()) };
    assert_eq!(s, cnt_status::CNT_INVALID_ARGUMENT, "{}", last_error());
    assert!(lease.is_null());

    let other = CString::new("worker:other").unwrap();
    let pose = CString::new("pose").unwrap();
    let caps = [capability(&pose, 0)];
    let mut results = [cnt_capability_result::default()];
    let second = connect(&b.socket);
    // SAFETY: valid arguments; 300 isn't a device index.
    let s = unsafe {
        cnt_register_worker(
            second,
            other.as_ptr(),
            ptr::null(),
            300,
            0,
            caps.as_ptr(),
            1,
            results.as_mut_ptr(),
            ptr::null_mut(),
            0,
        )
    };
    assert_eq!(s, cnt_status::CNT_INVALID_ARGUMENT);
    assert!(last_error().contains("device_index"), "{}", last_error());
    // SAFETY: live clients, not used again.
    unsafe {
        cnt_client_free(second);
        cnt_client_free(client);
    }
}

#[test]
fn an_inline_ingest_is_accepted() {
    let b = broker();
    let client = connect(&b.socket);
    let (name, shape) = (CString::new("events").unwrap(), CString::new("event").unwrap());
    let stream = register(client, &spec(&name, &shape)).unwrap().stream_id;
    let header = output_header(stream, 3);
    let bytes = br#"{"kind":"door"}"#;
    let at = cnt_location {
        kind: CNT_LOCATION_INLINE,
        inline_bytes: bytes.as_ptr(),
        inline_len: bytes.len(),
        arena: [0; CNT_NAME_MAX],
        arena_generation: 0,
        slab_class: 0,
        slot: 0,
        slot_generation: 0,
        offset: 0,
        length: 0,
    };
    // SAFETY: valid arguments.
    let s = unsafe { cnt_ingest(client, header.as_bytes().as_ptr(), header.as_bytes().len(), &raw const at) };
    assert_eq!(s, cnt_status::CNT_OK, "{}", last_error());
    // An accepted notice gets no reply; a refused one would arrive as an event.
    // SAFETY: an all-zero event is a valid value to overwrite.
    let mut e: cnt_event = unsafe { std::mem::zeroed() };
    // SAFETY: valid arguments.
    assert_eq!(unsafe { cnt_next_event(client, 200, &raw mut e) }, cnt_status::CNT_TIMEOUT);
    // SAFETY: a live client, not used again.
    unsafe { cnt_client_free(client) };
}

// ---- The data side: headers, slots, queries, promises, subscriptions, and jobs ----

fn header_spec(id: [u8; 16], stream: u32, start_ns: u64, shape: &CString, parents: &[u8]) -> cnt_header_spec {
    cnt_header_spec {
        id,
        stream_id: stream,
        start_ns,
        end_ns: start_ns + 100,
        seq: 0,
        layer: 0,
        format: 0,
        shape_name: shape.as_ptr(),
        shape_version: 1,
        parents: if parents.is_empty() { ptr::null() } else { parents.as_ptr() },
        n_parents: parents.len() / 16,
        source_tier: 0,
    }
}

fn build(spec: &cnt_header_spec) -> Vec<u8> {
    let mut out = vec![0u8; 512];
    let mut len = 0;
    // SAFETY: valid arguments.
    let s = unsafe { cnt_header_build(spec, out.as_mut_ptr(), out.len(), &raw mut len) };
    assert_eq!(s, cnt_status::CNT_OK, "{}", last_error());
    out.truncate(len);
    out
}

fn read(header: &[u8]) -> cnt_header_info {
    // SAFETY: an all-zero struct is a valid value to overwrite.
    let mut info: cnt_header_info = unsafe { std::mem::zeroed() };
    // SAFETY: valid arguments.
    let s = unsafe { cnt_header_read(header.as_ptr(), header.len(), &raw mut info) };
    assert_eq!(s, cnt_status::CNT_OK, "{}", last_error());
    info
}

fn new_id() -> [u8; 16] {
    let mut id = [0u8; 16];
    // SAFETY: writable for 16 bytes.
    unsafe { cnt_new_id(id.as_mut_ptr()) };
    id
}

#[test]
fn a_header_builds_and_reads_back_and_a_short_buffer_says_the_size_needed() {
    let shape = CString::new("pose").unwrap();
    let (a, b) = (new_id(), new_id());
    assert_ne!(a, b);
    let parents = [a, b].concat();
    let mut spec = header_spec(new_id(), 7, 1_000, &shape, &parents);
    (spec.seq, spec.layer, spec.format) = (3, 1, 4);

    let mut tiny = [0u8; 8];
    let mut needed = 0;
    // SAFETY: valid arguments.
    let s = unsafe { cnt_header_build(&raw const spec, tiny.as_mut_ptr(), tiny.len(), &raw mut needed) };
    assert_eq!(s, cnt_status::CNT_INVALID_ARGUMENT);
    assert!(needed > tiny.len(), "the size needed: {needed}");

    let header = build(&spec);
    assert_eq!(header.len(), needed);
    let info = read(&header);
    assert_eq!((info.id, info.stream_id, info.start_ns, info.end_ns), (spec.id, 7, 1_000, 1_100));
    assert_eq!((info.seq, info.layer, info.format, info.n_parents), (3, 1, 4, 2));

    spec.layer = 200;
    let mut len = 0;
    // SAFETY: valid arguments.
    let s = unsafe { cnt_header_build(&raw const spec, tiny.as_mut_ptr(), tiny.len(), &raw mut len) };
    assert_eq!(s, cnt_status::CNT_INVALID_ARGUMENT);
    assert!(last_error().contains("layer"), "{}", last_error());
}

/// Registers a sensor stream "cam" on `client`; returns its id.
fn cam(client: *mut cnt_client) -> u32 {
    let (name, shape) = (CString::new("cam").unwrap(), CString::new("frame").unwrap());
    let mut s = spec(&name, &shape);
    (s.kind, s.layer) = (CNT_SENSOR, 0);
    register(client, &s).unwrap().stream_id
}

fn subscribe(client: *mut cnt_client, stream: &CString) -> *mut cnt_subscription {
    let names = [stream.as_ptr()];
    let spec = cnt_subscribe_spec {
        streams: names.as_ptr(),
        n_streams: 1,
        start: CNT_START_LIVE,
        start_ns: 0,
        durable_name: ptr::null(),
        pinned: 0,
    };
    let mut sub = ptr::null_mut();
    // SAFETY: valid arguments.
    let s = unsafe { cnt_subscribe(client, &raw const spec, &raw mut sub) };
    assert_eq!(s, cnt_status::CNT_OK, "{}", last_error());
    sub
}

/// The next payload a subscription delivers, as its item list (skipping catch-up markers).
fn next_item(sub: *mut cnt_subscription) -> (u64, *mut cnt_items) {
    loop {
        // SAFETY: an all-zero struct is a valid value to overwrite.
        let mut e: cnt_sub_event = unsafe { std::mem::zeroed() };
        // SAFETY: valid arguments.
        let s = unsafe { cnt_subscription_next(sub, 5_000, &raw mut e) };
        assert_eq!(s, cnt_status::CNT_OK, "{}", last_error());
        if e.r#type == CNT_SUB_EVENT {
            return (e.position, e.item);
        }
    }
}

fn item(items: *const cnt_items, index: usize) -> cnt_item {
    // SAFETY: an all-zero struct is a valid value to overwrite.
    let mut out: cnt_item = unsafe { std::mem::zeroed() };
    // SAFETY: valid arguments.
    let s = unsafe { cnt_items_at(items, index, &raw mut out) };
    assert_eq!(s, cnt_status::CNT_OK, "{}", last_error());
    out
}

/// Writes `data` into a claimed slot of arena 0 and publishes it.
fn publish(client: *mut cnt_client, data: &[u8]) -> cnt_location {
    let mut slot = ptr::null_mut();
    // SAFETY: valid arguments.
    let s = unsafe { cnt_claim_slot(client, 0, 0, 0, &raw mut slot) };
    assert_eq!(s, cnt_status::CNT_OK, "{}", last_error());
    // SAFETY: a live claimed slot.
    assert!(unsafe { cnt_write_slot_capacity(slot) } >= data.len());
    // SAFETY: the slot's bytes hold at least data.len().
    unsafe { ptr::copy_nonoverlapping(data.as_ptr(), cnt_write_slot_data(slot), data.len()) };
    // SAFETY: an all-zero struct is a valid value to overwrite.
    let mut at: cnt_location = unsafe { std::mem::zeroed() };
    // SAFETY: valid arguments; the slot isn't used again.
    let s = unsafe { cnt_write_slot_publish(slot, data.len(), &raw mut at) };
    assert_eq!(s, cnt_status::CNT_OK, "{}", last_error());
    at
}

/// Ingests three frames over [100, 400) on cam from Tier 0 slots and waits for their commits.
fn frames(client: *mut cnt_client, cam: u32) {
    let (name, shape) = (CString::new("cam").unwrap(), CString::new("frame").unwrap());
    let sub = subscribe(client, &name);
    for n in 0..3u64 {
        let at = publish(client, format!("frame {n}").as_bytes());
        let header = build(&header_spec(new_id(), cam, 100 + 100 * n, &shape, &[]));
        // SAFETY: valid arguments.
        let s = unsafe { cnt_ingest(client, header.as_ptr(), header.len(), &raw const at) };
        assert_eq!(s, cnt_status::CNT_OK, "{}", last_error());
    }
    for _ in 0..3 {
        // SAFETY: a list the test owns.
        unsafe { cnt_items_free(next_item(sub).1) };
    }
    // SAFETY: a live subscription, not used again.
    unsafe { cnt_subscription_free(sub) };
}

#[test]
fn a_published_slot_is_ingested_delivered_found_by_a_query_and_read_in_place() {
    let b = broker();
    let client = connect(&b.socket);
    let cam_id = cam(client);
    let name = CString::new("cam").unwrap();
    let sub = subscribe(client, &name);

    // A claim given back unpublished, then one too short for what's published: refused, and freed either way.
    let mut slot = ptr::null_mut();
    // SAFETY: valid arguments.
    assert_eq!(unsafe { cnt_claim_slot(client, 0, 0, 0, &raw mut slot) }, cnt_status::CNT_OK);
    // SAFETY: a live claimed slot, not used again.
    unsafe { cnt_write_slot_abandon(slot) };
    // SAFETY: valid arguments.
    assert_eq!(unsafe { cnt_claim_slot(client, 0, 0, 0, &raw mut slot) }, cnt_status::CNT_OK);
    // SAFETY: an all-zero struct is a valid value to overwrite.
    let mut at: cnt_location = unsafe { std::mem::zeroed() };
    // SAFETY: valid arguments; the slot isn't used again.
    let too_long = unsafe { cnt_write_slot_publish(slot, usize::MAX, &raw mut at) };
    assert_eq!(too_long, cnt_status::CNT_INVALID_ARGUMENT);
    // SAFETY: valid arguments.
    assert_eq!(unsafe { cnt_claim_slot(client, 9, 0, 0, &raw mut slot) }, cnt_status::CNT_NOT_FOUND);

    let at = publish(client, b"hello");
    assert_eq!((at.kind, at.length), (CNT_LOCATION_SHM, 5));
    let shape = CString::new("frame").unwrap();
    let id = new_id();
    let header = build(&header_spec(id, cam_id, 500, &shape, &[]));
    // SAFETY: valid arguments.
    assert_eq!(unsafe { cnt_ingest(client, header.as_ptr(), header.len(), &raw const at) }, cnt_status::CNT_OK);

    let (position, delivered) = next_item(sub);
    assert!(position > 0);
    // SAFETY: a live list.
    assert_eq!(unsafe { cnt_items_count(delivered) }, 1);
    // SAFETY: the item's header is valid while the list lives.
    let got = unsafe { std::slice::from_raw_parts(item(delivered, 0).header, item(delivered, 0).header_len) };
    assert_eq!(read(got).id, id);
    // SAFETY: valid arguments, then lists the test owns.
    unsafe {
        assert_eq!(cnt_subscription_ack(sub, position), cnt_status::CNT_OK);
        cnt_items_free(delivered);
        cnt_subscription_free(sub);
    }

    let stream = CString::new("cam").unwrap();
    let q = cnt_query_spec {
        stream: stream.as_ptr(),
        range_start_ns: 0,
        range_end_ns: 1_000,
        mode: 0,
        band: 2,
        latency_budget_ms: 0,
        max_items: 0,
    };
    let mut result = ptr::null_mut();
    // SAFETY: valid arguments.
    assert_eq!(unsafe { cnt_query(client, &raw const q, &raw mut result) }, cnt_status::CNT_OK, "{}", last_error());
    // SAFETY: a live result.
    let items = unsafe { cnt_query_items(result) };
    // SAFETY: a live list.
    assert_eq!(unsafe { cnt_items_count(items) }, 1);
    let found = item(items, 0);
    assert_eq!(
        (found.at.kind, found.at.slot, found.at.slot_generation),
        (CNT_LOCATION_SHM, at.slot, at.slot_generation)
    );
    // SAFETY: an all-zero struct is a valid value to overwrite.
    let mut quote: cnt_promise_quote = unsafe { std::mem::zeroed() };
    // SAFETY: valid arguments.
    assert_eq!(unsafe { cnt_query_quote(result, &raw mut quote) }, cnt_status::CNT_NOT_FOUND, "hit-only: no quote");

    // Read in place under a pin.
    let slot_ref =
        cnt_slot_ref { slab_class: found.at.slab_class, index: found.at.slot, generation: found.at.slot_generation };
    let mut lease = ptr::null_mut();
    // SAFETY: valid arguments.
    assert_eq!(unsafe { cnt_lease_slot(client, 0, slot_ref, 1_000, &raw mut lease) }, cnt_status::CNT_OK);
    // SAFETY: the leased bytes are valid while the lease lives.
    let bytes = unsafe { std::slice::from_raw_parts(cnt_lease_data(lease), cnt_lease_len(lease)) };
    assert_eq!(bytes, b"hello");
    // SAFETY: handles the test owns, not used again.
    unsafe {
        assert_eq!(cnt_lease_release(lease), cnt_status::CNT_OK);
        cnt_query_result_free(result);
        cnt_client_free(client);
    }
}

/// A client with "cam" (three frames over [100, 400) in Tier 0) and a worker computing "pose" from it; returns the
/// client and pose's id.
fn pose_worker(socket: &CString) -> (*mut cnt_client, u32) {
    let client = connect(socket);
    let cam_id = cam(client);
    let (pose, shape) = (CString::new("pose").unwrap(), CString::new("pose").unwrap());
    let pose_id = register(client, &spec(&pose, &shape)).unwrap().stream_id;
    let input = CString::new("cam").unwrap();
    let needs = [cnt_requirement { stream: input.as_ptr(), before_ns: 0, after_ns: 0 }];
    let mut cap = capability(&pose, 0);
    (cap.inputs, cap.n_inputs) = (needs.as_ptr(), 1);
    let mut results = [cnt_capability_result::default()];
    let id = CString::new("worker:pose").unwrap();
    // SAFETY: valid arguments.
    let s = unsafe {
        cnt_register_worker(
            client,
            id.as_ptr(),
            ptr::null(),
            -1,
            0,
            &raw const cap,
            1,
            results.as_mut_ptr(),
            ptr::null_mut(),
            0,
        )
    };
    assert_eq!(s, cnt_status::CNT_OK, "{}", last_error());
    assert_eq!((results[0].accepted, results[0].runnable), (1, 1));
    frames(client, cam_id);
    (client, pose_id)
}

/// A quoted query for poses over the frames: the result (nothing found) and its quote.
fn quoted(consumer: *mut cnt_client) -> (*mut cnt_query_result, cnt_promise_quote) {
    let stream = CString::new("pose").unwrap();
    let q = cnt_query_spec {
        stream: stream.as_ptr(),
        range_start_ns: 100,
        range_end_ns: 400,
        mode: 1,
        band: 2,
        latency_budget_ms: 0,
        max_items: 0,
    };
    let mut result = ptr::null_mut();
    // SAFETY: valid arguments.
    assert_eq!(unsafe { cnt_query(consumer, &raw const q, &raw mut result) }, cnt_status::CNT_OK, "{}", last_error());
    // SAFETY: an all-zero struct is a valid value to overwrite.
    let mut quote: cnt_promise_quote = unsafe { std::mem::zeroed() };
    // SAFETY: valid arguments.
    assert_eq!(unsafe { cnt_query_quote(result, &raw mut quote) }, cnt_status::CNT_OK, "{}", last_error());
    (result, quote)
}

/// The next job dispatched to `client`.
fn next_job(client: *mut cnt_client) -> *mut cnt_job {
    loop {
        // SAFETY: an all-zero struct is a valid value to overwrite.
        let mut e: cnt_event = unsafe { std::mem::zeroed() };
        // SAFETY: valid arguments.
        assert_eq!(unsafe { cnt_next_event(client, 5_000, &raw mut e) }, cnt_status::CNT_OK, "{}", last_error());
        if e.r#type == CNT_EVENT_DISPATCH_JOB {
            return e.job;
        }
    }
}

fn commit_promise(consumer: *mut cnt_client, quote: &cnt_promise_quote) -> (u8, *mut cnt_promise) {
    let (mut life, mut promise) = (0u8, ptr::null_mut());
    // SAFETY: valid arguments.
    let s = unsafe { cnt_commit_promise(consumer, quote.promise_id.as_ptr(), 0, 2, &raw mut life, &raw mut promise) };
    assert_eq!(s, cnt_status::CNT_OK, "{}", last_error());
    (life, promise)
}

#[test]
fn a_c_worker_serves_the_job_a_committed_promise_dispatched() {
    let b = broker();
    let (worker, pose_id) = pose_worker(&b.socket);
    let consumer = connect(&b.socket);
    let (result, quote) = quoted(consumer);
    // SAFETY: a live result.
    assert_eq!(unsafe { cnt_items_count(cnt_query_items(result)) }, 0);
    // SAFETY: a live result.
    assert_eq!(unsafe { cnt_query_missing_count(result) }, 1);
    let (life, promise) = commit_promise(consumer, &quote);
    assert_eq!(life, 1, "Dispatching");

    let job = next_job(worker);
    // SAFETY: an all-zero struct is a valid value to overwrite.
    let mut info: cnt_job_info = unsafe { std::mem::zeroed() };
    // SAFETY: valid arguments.
    assert_eq!(unsafe { cnt_job_describe(job, &raw mut info) }, cnt_status::CNT_OK);
    assert_eq!((info.n_promises, info.has_lease, info.output_start_ns, info.output_end_ns), (1, 1, 100, 400));
    let mut served = [0u8; 16];
    // SAFETY: valid arguments.
    assert_eq!(unsafe { cnt_job_promise_at(job, 0, served.as_mut_ptr()) }, cnt_status::CNT_OK);
    assert_eq!(served, quote.promise_id);
    // SAFETY: a live job.
    let inputs = unsafe { cnt_job_inputs(job) };
    // SAFETY: a live list.
    let n = unsafe { cnt_items_count(inputs) };
    assert_eq!(n, 3);
    let parents: Vec<u8> = (0..n)
        .flat_map(|i| {
            let it = item(inputs, i);
            assert_eq!(it.at.kind, CNT_LOCATION_SHM, "inputs are in Tier 0");
            // SAFETY: the header is valid while the job lives.
            read(unsafe { std::slice::from_raw_parts(it.header, it.header_len) }).id
        })
        .collect();

    let mut lease = ptr::null_mut();
    // SAFETY: valid arguments.
    assert_eq!(unsafe { cnt_job_take_lease(worker, job, &raw mut lease) }, cnt_status::CNT_OK, "{}", last_error());
    let mut again = ptr::null_mut();
    // SAFETY: valid arguments.
    assert_eq!(unsafe { cnt_job_take_lease(worker, job, &raw mut again) }, cnt_status::CNT_PROTOCOL, "once per job");

    let shape = CString::new("pose").unwrap();
    let out_id = new_id();
    let mut spec = header_spec(out_id, pose_id, 100, &shape, &parents);
    (spec.end_ns, spec.layer, spec.format) = (400, 1, 0);
    let header = build(&spec);
    let body = b"pose";
    // SAFETY: an all-zero struct is a valid value to overwrite.
    let mut at: cnt_location = unsafe { std::mem::zeroed() };
    (at.kind, at.inline_bytes, at.inline_len) = (CNT_LOCATION_INLINE, body.as_ptr(), body.len());
    let output = cnt_output { header: header.as_ptr(), header_len: header.len(), at: &raw const at };
    // SAFETY: valid arguments; the lease isn't used again after an OK.
    let s = unsafe { cnt_work_lease_commit(lease, &raw const output, 1, ptr::null(), ptr::null()) };
    assert_eq!(s, cnt_status::CNT_OK, "{}", last_error());

    let (mut part, mut last) = (ptr::null_mut(), 0u8);
    // SAFETY: valid arguments.
    let s = unsafe { cnt_promise_next_part(promise, 5_000, &raw mut part, &raw mut last) };
    assert_eq!(s, cnt_status::CNT_OK, "{}", last_error());
    assert_eq!(last, 1);
    // SAFETY: a live list.
    assert_eq!(unsafe { cnt_items_count(part) }, 1);
    let got = item(part, 0);
    // SAFETY: the header is valid while the list lives.
    assert_eq!(read(unsafe { std::slice::from_raw_parts(got.header, got.header_len) }).id, out_id);
    // SAFETY: a list the test owns, not used again.
    unsafe { cnt_items_free(part) };
    // SAFETY: valid arguments.
    let after = unsafe { cnt_promise_next_part(promise, 100, &raw mut part, &raw mut last) };
    assert_eq!(after, cnt_status::CNT_CLOSED, "the final part was taken");
    // SAFETY: handles the test owns, not used again.
    unsafe {
        cnt_job_free(job);
        cnt_promise_free(promise);
        cnt_query_result_free(result);
        cnt_client_free(consumer);
        cnt_client_free(worker);
    }
}

#[test]
fn cancelling_a_promise_ends_its_held_handle_and_a_second_cancel_is_not_found() {
    let b = broker();
    let (worker, _) = pose_worker(&b.socket);
    let consumer = connect(&b.socket);
    let (result, quote) = quoted(consumer);
    let (_, promise) = commit_promise(consumer, &quote);
    let job = next_job(worker);

    let mut life = 0u8;
    // SAFETY: valid arguments.
    let s = unsafe { cnt_cancel_promise(consumer, quote.promise_id.as_ptr(), &raw mut life) };
    assert_eq!((s, life), (cnt_status::CNT_OK, 2), "Cancelled: {}", last_error());
    let (mut part, mut last) = (ptr::null_mut(), 0u8);
    // SAFETY: valid arguments.
    let ended = unsafe { cnt_promise_next_part(promise, 5_000, &raw mut part, &raw mut last) };
    assert_eq!(ended, cnt_status::CNT_CLOSED);
    assert!(part.is_null());
    // SAFETY: valid arguments.
    let gone = unsafe { cnt_cancel_promise(consumer, quote.promise_id.as_ptr(), ptr::null_mut()) };
    assert_eq!(gone, cnt_status::CNT_BROKER);
    assert_eq!(cnt_last_broker_code(), fb::ErrorCode::NotFound.0);
    // SAFETY: handles the test owns, not used again.
    unsafe {
        cnt_job_free(job);
        cnt_promise_free(promise);
        cnt_query_result_free(result);
        cnt_client_free(consumer);
        cnt_client_free(worker);
    }
}
