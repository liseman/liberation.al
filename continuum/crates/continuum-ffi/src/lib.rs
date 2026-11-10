//! The C ABI over continuum-client: what C and C++ workers (SLAM, odometry, the OAK-D's DepthAI worker) link to.
//! `include/continuum.h` is generated from this file by cbindgen (`mise run continuum-ffi-header`), and
//! `include/continuum.hpp` wraps it for C++20 with RAII leases and exceptions.
//!
//! The rules every function here follows:
//! - **Opaque handles.** `cnt_client`, `cnt_lease`, `cnt_work_lease`, and the data side's handles (`cnt_write_slot`,
//!   `cnt_items`, `cnt_query_result`, `cnt_promise`, `cnt_subscription`, `cnt_job`) are only ever pointers to C;
//!   nothing Rust-shaped crosses.
//! - **Status codes.** Every fallible function returns a `cnt_status` (`CNT_OK` is 0). The message for the last
//!   failure on the calling thread is `cnt_last_error_message()`, and the broker's own error code (for
//!   `CNT_BROKER`) is `cnt_last_broker_code()`.
//! - **No unwinding into C.** Every function catches a panic and returns `CNT_PANIC` instead, and stays safe to call
//!   during process exit (from an atexit handler or a C++ static destructor), when it just can't record a message.
//! - **Explicit ownership.** A handle the library hands out belongs to the caller until it's passed to its own
//!   `_free` (or `cnt_lease_release`, `cnt_write_slot_publish`, and the like); one it lends (a query's items, a job's
//!   inputs) says so and lives as long as what lent it. Strings passed in are borrowed for the call only. Strings and
//!   pointers handed out say how long they live: a lease's data until the lease is released; an arena's base while
//!   the client lives; the last error message until the next call on the same thread. Names come back in
//!   fixed-size arrays in the caller's structs, so nothing needs freeing.
//! - **Blocking calls.** Each client owns a small tokio runtime; a call blocks until the broker answers or the
//!   request timeout passes. A client can be used from one thread at a time (wrap it in a mutex to share it).
//! - **Versioned.** `cnt_abi_version()` changes whenever a struct's layout or a function's meaning changes; new
//!   functions and new status codes are added without changing it.

#![allow(non_camel_case_types)] // C naming, for the C header

mod data;
mod header;
mod worker;
pub use data::*;
pub use header::*;
pub use worker::*;

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::ptr;
use std::time::Duration;

use continuum_client::{Client, ClientOptions, Error as ClientError, StreamSpec, StreamStorage};
use continuum_core::SlotHandle;
use continuum_core::schema::continuum as fb;
use continuum_shm::Lease;

/// The ABI version. It changes when a struct's layout or a function's meaning changes.
pub const CNT_ABI_VERSION: u32 = 2;

/// The longest name the ABI hands back (a shared memory name is at most 31 bytes, plus the NUL).
pub const CNT_NAME_MAX: usize = 32;

/// What a call returned.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum cnt_status {
    /// It worked.
    CNT_OK = 0,
    /// An argument was null, not UTF-8, or out of range.
    CNT_INVALID_ARGUMENT = 1,
    /// The socket or the operating system failed.
    CNT_IO = 2,
    /// The broker refused the request; `cnt_last_broker_code()` says why.
    CNT_BROKER = 3,
    /// The broker sent something the protocol doesn't allow.
    CNT_PROTOCOL = 4,
    /// No reply within the request timeout.
    CNT_TIMEOUT = 5,
    /// The connection is closed.
    CNT_CLOSED = 6,
    /// An arena couldn't be mapped or doesn't match its description.
    CNT_ARENA = 7,
    /// The slot was reclaimed or reused, or the lease lapsed.
    CNT_STALE = 8,
    /// The slot holds the most pins it can.
    CNT_BUSY = 9,
    /// No such arena, slot, or ring.
    CNT_NOT_FOUND = 10,
    /// The broker didn't grant a lease now; try again later (`cnt_quote` says when).
    CNT_DENIED = 11,
    /// A bug: the library panicked. The message says where.
    CNT_PANIC = 99,
}

thread_local! {
    static LAST_ERROR: RefCell<(CString, u16)> = RefCell::new((CString::default(), 0));
}

/// Records the outcome for `cnt_last_error_message`. Never panics: during process exit the thread's storage may
/// already be gone (a C++ static destructor or an atexit handler calling in), and then nothing is recorded.
pub(crate) fn set_error(status: cnt_status, message: &str, broker_code: u16) -> cnt_status {
    let message = CString::new(message.replace('\0', " ")).unwrap_or_default();
    let _ = LAST_ERROR.try_with(|e| {
        if let Ok(mut e) = e.try_borrow_mut() {
            *e = (message, broker_code);
        }
    });
    status
}

pub(crate) fn from_client(e: &ClientError) -> cnt_status {
    let (status, code) = match e {
        ClientError::Io(_) | ClientError::Frame(_) => (cnt_status::CNT_IO, 0),
        ClientError::Broker { code, .. } => (cnt_status::CNT_BROKER, code.0),
        ClientError::Protocol(_) => (cnt_status::CNT_PROTOCOL, 0),
        ClientError::Arena { .. } => (cnt_status::CNT_ARENA, 0),
        ClientError::Timeout(_) => (cnt_status::CNT_TIMEOUT, 0),
        ClientError::Closed => (cnt_status::CNT_CLOSED, 0),
        // A failed promise is the broker's answer (its reason is the message).
        ClientError::PromiseFailed { .. } => (cnt_status::CNT_BROKER, 0),
    };
    set_error(status, &e.to_string(), code)
}

pub(crate) fn from_shm(e: &continuum_shm::Error) -> cnt_status {
    use continuum_shm::Error as E;
    let status = match e {
        E::Stale | E::Expired => cnt_status::CNT_STALE,
        E::Busy => cnt_status::CNT_BUSY,
        E::NoSuchSlot | E::NoSuchRing => cnt_status::CNT_NOT_FOUND,
        E::BadLease(_) => cnt_status::CNT_INVALID_ARGUMENT,
        E::Io(_) => cnt_status::CNT_IO,
        _ => cnt_status::CNT_ARENA,
    };
    set_error(status, &e.to_string(), 0)
}

pub(crate) fn invalid(message: &str) -> cnt_status {
    set_error(cnt_status::CNT_INVALID_ARGUMENT, message, 0)
}

/// Runs `f`, turning a panic into `CNT_PANIC` and recording success or failure for `cnt_last_error_message`.
pub(crate) fn guard(f: impl FnOnce() -> cnt_status) -> cnt_status {
    let ran = catch_unwind(AssertUnwindSafe(|| {
        let status = f();
        if status == cnt_status::CNT_OK {
            set_error(cnt_status::CNT_OK, "", 0);
        }
        status
    }));
    match ran {
        Ok(status) => status,
        Err(panic) => {
            // Recording it can't unwind into C either (a payload whose drop panics, say).
            catch_unwind(AssertUnwindSafe(move || {
                let what = panic
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_owned())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic".into());
                set_error(cnt_status::CNT_PANIC, &format!("continuum-ffi panicked: {what}"), 0)
            }))
            .unwrap_or(cnt_status::CNT_PANIC)
        }
    }
}

/// A borrowed C string as `&str`, or `None` for null or non-UTF-8.
///
/// # Safety
///
/// `s` is null or a valid NUL-terminated string that outlives `'a`.
pub(crate) unsafe fn str_arg<'a>(s: *const c_char) -> Option<&'a str> {
    if s.is_null() {
        return None;
    }
    // SAFETY: as the caller promises.
    unsafe { CStr::from_ptr(s) }.to_str().ok()
}

/// Copies `s` into a fixed-size, NUL-terminated name field, truncating to fit.
pub(crate) fn copy_name(s: &str, out: &mut [c_char; CNT_NAME_MAX]) {
    out.fill(0);
    for (dst, &b) in out.iter_mut().zip(s.as_bytes().iter().take(CNT_NAME_MAX - 1)) {
        *dst = b as c_char;
    }
}

// ---- Versions and errors ----

/// The ABI version this library implements (`CNT_ABI_VERSION` when it was built). Check it against the header's.
#[unsafe(no_mangle)]
pub extern "C" fn cnt_abi_version() -> u32 {
    CNT_ABI_VERSION
}

/// The library's version, like "0.1.0". A static string; don't free it.
#[unsafe(no_mangle)]
pub extern "C" fn cnt_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// What went wrong in the last call on this thread that failed ("" after a call that worked). Valid until the next
/// call on this thread; don't free it.
#[unsafe(no_mangle)]
pub extern "C" fn cnt_last_error_message() -> *const c_char {
    LAST_ERROR.try_with(|e| e.try_borrow().map(|e| e.0.as_ptr()).ok()).ok().flatten().unwrap_or(c"".as_ptr())
}

/// The broker's error code (the schema's `ErrorCode`) after a call returned `CNT_BROKER` on this thread, else 0.
#[unsafe(no_mangle)]
pub extern "C" fn cnt_last_broker_code() -> u16 {
    LAST_ERROR.try_with(|e| e.try_borrow().map(|e| e.1).unwrap_or(0)).unwrap_or(0)
}

// ---- Connecting ----

/// A connection to the broker. Opaque: made by `cnt_connect`, freed by `cnt_client_free`.
pub struct cnt_client {
    // Field order is drop order: the client (and its tasks) before the runtime they run on.
    pub(crate) client: Client,
    pub(crate) runtime: tokio::runtime::Runtime,
}

/// How to connect. Start from `cnt_connect_options_default`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cnt_connect_options {
    /// The name the broker logs the session under (borrowed for the call). Null uses "c".
    pub client_name: *const c_char,
    /// How often the client heartbeats, in milliseconds. At least 10.
    pub heartbeat_interval_ms: u32,
    /// How long a call waits for the broker's reply, in milliseconds. At least 10.
    pub request_timeout_ms: u32,
}

/// The default options: name "c", a heartbeat every 5 s, and a 10 s request timeout.
#[unsafe(no_mangle)]
pub extern "C" fn cnt_connect_options_default() -> cnt_connect_options {
    cnt_connect_options { client_name: ptr::null(), heartbeat_interval_ms: 5_000, request_timeout_ms: 10_000 }
}

/// Connects to the broker's Unix socket, says Hello, and maps the arenas Welcome hands over. On `CNT_OK`, `*out` is
/// a client the caller owns (free it with `cnt_client_free`); on failure it's set to null.
///
/// # Safety
///
/// `socket` is a valid NUL-terminated path. `options` is null (for the defaults) or points to valid options.
/// `out` points to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_connect(
    socket: *const c_char,
    options: *const cnt_connect_options,
    out: *mut *mut cnt_client,
) -> cnt_status {
    guard(|| {
        if out.is_null() {
            return invalid("out is null");
        }
        // SAFETY: out is non-null and writable, as the caller promises.
        unsafe { *out = ptr::null_mut() };
        // SAFETY: as the caller promises.
        let Some(socket) = (unsafe { str_arg(socket) }) else { return invalid("socket is null or not UTF-8") };
        let options = if options.is_null() {
            cnt_connect_options_default()
        } else {
            // SAFETY: non-null, and valid as the caller promises.
            unsafe { *options }
        };
        if options.heartbeat_interval_ms < 10 || options.request_timeout_ms < 10 {
            return invalid("heartbeat_interval_ms and request_timeout_ms must be at least 10");
        }
        let name = if options.client_name.is_null() {
            "c"
        } else {
            // SAFETY: non-null, and a valid string as the caller promises.
            match unsafe { str_arg(options.client_name) } {
                Some(n) => n,
                None => return invalid("client_name is not UTF-8"),
            }
        };
        let opts = ClientOptions {
            client_name: name.to_owned(),
            heartbeat_interval: Duration::from_millis(options.heartbeat_interval_ms.into()),
            request_timeout: Duration::from_millis(options.request_timeout_ms.into()),
            subscription_overflow: continuum_client::SubscriptionOverflow::Close,
        };
        let runtime = match tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("continuum-ffi")
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => return set_error(cnt_status::CNT_IO, &format!("couldn't start the runtime: {e}"), 0),
        };
        match runtime.block_on(Client::connect(Path::new(socket), opts)) {
            Ok(client) => {
                let handle = Box::new(cnt_client { client, runtime });
                // SAFETY: out is writable, checked above.
                unsafe { *out = Box::into_raw(handle) };
                cnt_status::CNT_OK
            }
            Err(e) => from_client(&e),
        }
    })
}

/// Says Goodbye: the broker drops the session's pins and leases. Returns once the Goodbye is written. Calls after
/// this fail with `CNT_CLOSED`; the client still has to be freed.
///
/// # Safety
///
/// `client` is a live client from `cnt_connect`, used by no other thread during the call. `reason` is null or a
/// valid NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_client_close(client: *mut cnt_client, reason: *const c_char) -> cnt_status {
    guard(|| {
        // SAFETY: as the caller promises.
        let Some(c) = (unsafe { client.as_ref() }) else { return invalid("client is null") };
        // SAFETY: as the caller promises.
        let reason = unsafe { str_arg(reason) }.unwrap_or("closed");
        match c.runtime.block_on(c.client.close(reason)) {
            Ok(()) => cnt_status::CNT_OK,
            Err(e) => from_client(&e),
        }
    })
}

/// Frees a client: closes the connection (without a Goodbye; call `cnt_client_close` first for a clean one) and
/// stops its runtime. Leases taken through it stay valid. Null is a no-op.
///
/// # Safety
///
/// `client` is null or a live client from `cnt_connect` that no other thread is using, and isn't used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_client_free(client: *mut cnt_client) {
    let _ = guard(|| {
        if !client.is_null() {
            // SAFETY: made by Box::into_raw in cnt_connect and not freed before, as the caller promises.
            drop(unsafe { Box::from_raw(client) });
        }
        cnt_status::CNT_OK
    });
}

/// Writes the session's id (a UUIDv7, 16 bytes in RFC 9562 order) to `out`.
///
/// # Safety
///
/// `client` is a live client; `out` points to 16 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_session_id(client: *const cnt_client, out: *mut u8) -> cnt_status {
    guard(|| {
        // SAFETY: as the caller promises.
        let Some(c) = (unsafe { client.as_ref() }) else { return invalid("client is null") };
        if out.is_null() {
            return invalid("out is null");
        }
        // SAFETY: out has room for 16 bytes, as the caller promises.
        unsafe { ptr::copy_nonoverlapping(c.client.session_id().0.as_ptr(), out, 16) };
        cnt_status::CNT_OK
    })
}

/// Sends a heartbeat and writes the broker's clock (UTC nanoseconds) to `*broker_now_ns` (if non-null). The client
/// also heartbeats on its own; this is to check the broker is responsive.
///
/// # Safety
///
/// `client` is a live client used by no other thread during the call; `broker_now_ns` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_heartbeat(client: *mut cnt_client, broker_now_ns: *mut u64) -> cnt_status {
    guard(|| {
        // SAFETY: as the caller promises.
        let Some(c) = (unsafe { client.as_ref() }) else { return invalid("client is null") };
        match c.runtime.block_on(c.client.heartbeat(None)) {
            Ok(now) => {
                if !broker_now_ns.is_null() {
                    // SAFETY: non-null and writable, as the caller promises.
                    unsafe { *broker_now_ns = now };
                }
                cnt_status::CNT_OK
            }
            Err(e) => from_client(&e),
        }
    })
}

// ---- Streams ----

// Input fields from C are plain integers checked here, never Rust enums: a C caller can put any value in a field, and
// reading an out-of-range value as a Rust enum is undefined behavior.

/// `cnt_stream_spec.kind`: captured by a device.
pub const CNT_SENSOR: u8 = 0;
/// `cnt_stream_spec.kind`: computed from other streams.
pub const CNT_DERIVED: u8 = 1;
/// `cnt_stream_spec.storage`: slots of `slab_class`.
pub const CNT_SLAB: u8 = 0;
/// `cnt_stream_spec.storage`: a ring of `ring_stride_bytes` records holding `ring_seconds` (0 for the default).
pub const CNT_RING: u8 = 1;

/// A stream to register. Start from zeros; strings are borrowed for the call.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cnt_stream_spec {
    /// The stream's name, like "cam0.depth".
    pub name: *const c_char,
    /// `CNT_SENSOR` or `CNT_DERIVED`.
    pub kind: u8,
    /// The payload format: the schema's `PayloadFormat` value (0 RawBytes, 1 DLPackTensor, 2 ArrowRecordBatch,
    /// 3 EncodedVideo, 4 FlatBuffer, 5 Text).
    pub format: u8,
    /// The domain layer: the schema's `DomainLayer` value (0 Layer1Raw, 1 Layer2Decision, 2 Layer3Interpretation).
    pub layer: u8,
    /// A raw format like "NV12", or a named shape.
    pub shape_name: *const c_char,
    /// The shape's version.
    pub shape_version: u32,
    /// Payloads per second, nominally (0 if irregular).
    pub nominal_rate_hz: f32,
    /// `CNT_SLAB` or `CNT_RING`.
    pub storage: u8,
    /// For `CNT_SLAB`: the size class.
    pub slab_class: u16,
    /// For `CNT_RING`: bytes per record.
    pub ring_stride_bytes: u32,
    /// For `CNT_RING`: seconds the ring holds (0 for the broker's default).
    pub ring_seconds: f32,
    /// Nonzero if the clock anchor below is set (sensors). (A byte, not a bool: any value from C is valid.)
    pub has_anchor: u8,
    /// The anchor's boot id (16 bytes).
    pub anchor_boot_id: [u8; 16],
    /// The anchor's monotonic reading, nanoseconds.
    pub anchor_monotonic_ns: u64,
    /// The UTC time it maps to, nanoseconds since the epoch.
    pub anchor_utc_ns: u64,
}

/// What `cnt_register_stream` returns.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cnt_stream_registered {
    /// The stream's id.
    pub stream_id: u32,
    /// Whether a standing lease came with it (a capture stream); the fields below are set only if so.
    pub has_lease: bool,
    /// The lease's id.
    pub lease_id: u64,
    /// The arena it's in (NUL-terminated).
    pub lease_arena: [c_char; CNT_NAME_MAX],
    /// That arena's generation.
    pub lease_arena_generation: u32,
    /// The first of the lease's slots.
    pub lease_first_slot: u32,
    /// How many slots.
    pub lease_slot_count: u32,
    /// Bytes per slot.
    pub lease_slot_bytes: u64,
    /// For a ring: its offset in the arena.
    pub lease_ring_offset: u64,
    /// For a ring: its record count.
    pub lease_ring_records: u32,
}

/// Registers a stream by name and writes its id (and standing lease, if any) to `*out`. Registering the same name
/// with the same kind, layer, format, and shape again returns the same id; a different one fails with `CNT_BROKER`
/// (code Conflict).
///
/// # Safety
///
/// `client` is a live client used by no other thread during the call; `spec` points to a valid spec whose strings
/// are valid NUL-terminated strings; `out` points to writable storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_register_stream(
    client: *mut cnt_client,
    spec: *const cnt_stream_spec,
    out: *mut cnt_stream_registered,
) -> cnt_status {
    guard(|| {
        // SAFETY: as the caller promises.
        let Some(c) = (unsafe { client.as_ref() }) else { return invalid("client is null") };
        // SAFETY: as the caller promises.
        let Some(spec) = (unsafe { spec.as_ref() }) else { return invalid("spec is null") };
        if out.is_null() {
            return invalid("out is null");
        }
        // SAFETY: the spec's strings are valid, as the caller promises.
        let (Some(name), Some(shape)) = (unsafe { str_arg(spec.name) }, unsafe { str_arg(spec.shape_name) }) else {
            return invalid("name or shape_name is null or not UTF-8");
        };
        let format = fb::PayloadFormat(spec.format);
        if format.variant_name().is_none() {
            return invalid("format isn't a PayloadFormat value");
        }
        let layer = fb::DomainLayer(spec.layer);
        if layer.variant_name().is_none() {
            return invalid("layer isn't a DomainLayer value");
        }
        let storage = match spec.storage {
            CNT_SLAB => StreamStorage::Slab { class: spec.slab_class },
            CNT_RING => StreamStorage::Ring { stride_bytes: spec.ring_stride_bytes, seconds: spec.ring_seconds },
            _ => return invalid("storage must be CNT_SLAB or CNT_RING"),
        };
        let kind = match spec.kind {
            CNT_SENSOR => fb::StreamKind::Sensor,
            CNT_DERIVED => fb::StreamKind::Derived,
            _ => return invalid("kind must be CNT_SENSOR or CNT_DERIVED"),
        };
        let anchor = (spec.has_anchor != 0).then(|| {
            fb::ClockAnchor::new(&fb::Uuid::new(&spec.anchor_boot_id), spec.anchor_monotonic_ns, spec.anchor_utc_ns)
        });
        let s = StreamSpec {
            name: name.to_owned(),
            kind,
            format,
            shape_name: shape.to_owned(),
            shape_version: spec.shape_version,
            nominal_rate_hz: spec.nominal_rate_hz,
            storage,
            anchor,
            layer,
        };
        match c.runtime.block_on(c.client.register_stream(&s)) {
            Ok(r) => {
                let mut o = cnt_stream_registered {
                    stream_id: r.stream_id,
                    has_lease: r.lease.is_some(),
                    lease_id: 0,
                    lease_arena: [0; CNT_NAME_MAX],
                    lease_arena_generation: 0,
                    lease_first_slot: 0,
                    lease_slot_count: 0,
                    lease_slot_bytes: 0,
                    lease_ring_offset: 0,
                    lease_ring_records: 0,
                };
                if let Some(l) = r.lease {
                    o.lease_id = l.lease_id;
                    copy_name(&l.arena, &mut o.lease_arena);
                    o.lease_arena_generation = l.arena_generation;
                    o.lease_first_slot = l.first_slot;
                    o.lease_slot_count = l.slot_count;
                    o.lease_slot_bytes = l.slot_bytes;
                    o.lease_ring_offset = l.ring_offset;
                    o.lease_ring_records = l.ring_records;
                }
                // SAFETY: out is non-null and writable, as the caller promises.
                unsafe { *out = o };
                cnt_status::CNT_OK
            }
            Err(e) => from_client(&e),
        }
    })
}

// ---- Arenas and leases ----

/// An arena the broker handed this client.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cnt_arena_info {
    /// Its name (NUL-terminated).
    pub name: [c_char; CNT_NAME_MAX],
    /// Its generation.
    pub generation: u32,
    /// Its size in bytes.
    pub size_bytes: u64,
    /// Where it's mapped in this process (16 KB aligned). Valid while the client or any of its leases lives.
    pub base: *const u8,
}

/// How many arenas the broker handed this client.
///
/// # Safety
///
/// `client` is null (0) or a live client.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_arena_count(client: *const cnt_client) -> usize {
    // SAFETY: as the caller promises.
    unsafe { client.as_ref() }.map_or(0, |c| c.client.arenas().len())
}

/// Writes arena `index` (in Welcome order) to `*out`. `CNT_NOT_FOUND` past the last one.
///
/// # Safety
///
/// `client` is a live client; `out` points to writable storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_arena_info_at(
    client: *const cnt_client,
    index: usize,
    out: *mut cnt_arena_info,
) -> cnt_status {
    guard(|| {
        // SAFETY: as the caller promises.
        let Some(c) = (unsafe { client.as_ref() }) else { return invalid("client is null") };
        if out.is_null() {
            return invalid("out is null");
        }
        let Some(a) = c.client.arenas().get(index) else {
            return set_error(cnt_status::CNT_NOT_FOUND, &format!("no arena {index}"), 0);
        };
        let mut info = cnt_arena_info {
            name: [0; CNT_NAME_MAX],
            generation: a.info.generation,
            size_bytes: a.info.size_bytes,
            base: a.arena.base_ptr(),
        };
        copy_name(&a.info.name, &mut info.name);
        // SAFETY: out is non-null and writable, as the caller promises.
        unsafe { *out = info };
        cnt_status::CNT_OK
    })
}

/// A slot in an arena: its size class, index, and generation (as a query, job, or ring record names it).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct cnt_slot_ref {
    /// The size class.
    pub slab_class: u16,
    /// The slot's index in its class.
    pub index: u32,
    /// The slot's generation.
    pub generation: u32,
}

/// A pin on a published slot: its bytes stay put until it's released (or its time runs out). Opaque: made by
/// `cnt_lease_slot`, freed by `cnt_lease_release`.
pub struct cnt_lease {
    lease: Lease,
}

/// Pins slot `slot` of arena `arena_index` for `ttl_ms` milliseconds (at most 60 000) and writes the lease to
/// `*out`, which the caller owns until `cnt_lease_release`. `CNT_STALE` if the slot was reclaimed or reused since
/// `slot` was made. The lease keeps the arena mapped, even after the client is freed.
///
/// # Safety
///
/// `client` is a live client; `out` points to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_lease_slot(
    client: *const cnt_client,
    arena_index: usize,
    slot: cnt_slot_ref,
    ttl_ms: u32,
    out: *mut *mut cnt_lease,
) -> cnt_status {
    guard(|| {
        if out.is_null() {
            return invalid("out is null");
        }
        // SAFETY: out is non-null and writable, as the caller promises.
        unsafe { *out = ptr::null_mut() };
        // SAFETY: as the caller promises.
        let Some(c) = (unsafe { client.as_ref() }) else { return invalid("client is null") };
        let Some(a) = c.client.arenas().get(arena_index) else {
            return set_error(cnt_status::CNT_NOT_FOUND, &format!("no arena {arena_index}"), 0);
        };
        let handle = SlotHandle { class: slot.slab_class, index: slot.index, generation: slot.generation };
        match a.arena.lease(handle, Duration::from_millis(ttl_ms.into())) {
            Ok(lease) => {
                // SAFETY: out is writable, checked above.
                unsafe { *out = Box::into_raw(Box::new(cnt_lease { lease })) };
                cnt_status::CNT_OK
            }
            Err(e) => from_shm(&e),
        }
    })
}

/// The leased bytes (16 KB aligned), read-only. Valid until the lease is released; check `cnt_lease_valid` after
/// reading if the reader may have run past the lease's time.
///
/// # Safety
///
/// `lease` is a live lease.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_lease_data(lease: *const cnt_lease) -> *const u8 {
    // SAFETY: as the caller promises.
    unsafe { lease.as_ref() }.map_or(ptr::null(), |l| l.lease.as_ptr())
}

/// The leased payload's length in bytes.
///
/// # Safety
///
/// `lease` is null (0) or a live lease.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_lease_len(lease: *const cnt_lease) -> usize {
    // SAFETY: as the caller promises.
    unsafe { lease.as_ref() }.map_or(0, |l| l.lease.len())
}

/// Whether the lease still holds its slot: not past its time and the slot not reused.
///
/// # Safety
///
/// `lease` is null (false) or a live lease.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_lease_valid(lease: *const cnt_lease) -> bool {
    // SAFETY: as the caller promises.
    unsafe { lease.as_ref() }.is_some_and(|l| l.lease.is_valid())
}

/// Releases a lease (the slot's pin goes) and frees it. `CNT_STALE` if it had already lapsed (it's freed either
/// way). Null is a no-op that returns `CNT_OK`.
///
/// # Safety
///
/// `lease` is null or a live lease from `cnt_lease_slot`, not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_lease_release(lease: *mut cnt_lease) -> cnt_status {
    guard(|| {
        if lease.is_null() {
            return cnt_status::CNT_OK;
        }
        // SAFETY: made by Box::into_raw in cnt_lease_slot and not freed before, as the caller promises.
        let l = unsafe { Box::from_raw(lease) };
        if l.lease.release() {
            cnt_status::CNT_OK
        } else {
            set_error(cnt_status::CNT_STALE, "the lease had lapsed or its slot was reused before release", 0)
        }
    })
}
