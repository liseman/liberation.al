//! The worker side of the C ABI: ingest, registering a worker, quoting for a lease, the lease itself (renew,
//! keep-alive, commit, abort, release), the voice turn, and the events the broker pushes. The same rules as the rest
//! of the ABI (see the crate docs): integers rather than enums for anything C passes in, explicit ownership, and a
//! status code from every call that can fail.

use std::ffi::{CStr, c_char};
use std::ptr;
use std::time::Duration;

use continuum_client::{
    BrokerEvent, CapabilitySpec, Cost, CoverageDone, Quote, QuoteSpec, Requirement, WorkLease, WorkerSpec,
};
use continuum_core::schema::continuum as fb;
use continuum_core::{HeaderBuf, StorageLocator, Uuid};

use crate::{CNT_NAME_MAX, cnt_client, cnt_status, from_client, guard, invalid, set_error, str_arg};

/// The most bytes of a reason `cnt_event` carries (NUL included); a longer one is cut.
pub const CNT_REASON_MAX: usize = 256;

/// `cnt_location.kind`: no bytes (a header-only payload).
pub const CNT_LOCATION_NONE: u8 = 0;
/// `cnt_location.kind`: the bytes inline (at most 4096).
pub const CNT_LOCATION_INLINE: u8 = 1;
/// `cnt_location.kind`: the bytes in a shared memory slot.
pub const CNT_LOCATION_SHM: u8 = 2;
/// `cnt_location.kind`, only in what the broker hands back: the bytes are in a chunk file or a ring, which this ABI
/// doesn't read yet.
pub const CNT_LOCATION_OTHER: u8 = 3;

/// Where a payload's bytes are, for `cnt_ingest` and a commit's outputs.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cnt_location {
    /// `CNT_LOCATION_NONE`, `CNT_LOCATION_INLINE`, or `CNT_LOCATION_SHM`.
    pub kind: u8,
    /// Inline: the bytes (borrowed for the call).
    pub inline_bytes: *const u8,
    /// Inline: how many.
    pub inline_len: usize,
    /// Shm: the arena's name (NUL-terminated).
    pub arena: [c_char; CNT_NAME_MAX],
    /// Shm: the arena's generation.
    pub arena_generation: u32,
    /// Shm: the slot's size class.
    pub slab_class: u16,
    /// Shm: the slot's index.
    pub slot: u32,
    /// Shm: the slot's generation.
    pub slot_generation: u32,
    /// Shm: the offset in the arena.
    pub offset: u64,
    /// Shm: the length in bytes.
    pub length: u64,
}

/// The owned locator a `cnt_location` describes.
///
/// # Safety
///
/// `loc` is null or valid, and its inline bytes are valid for `inline_len` bytes.
unsafe fn locator(loc: *const cnt_location) -> Result<Option<StorageLocator>, cnt_status> {
    // SAFETY: as the caller promises.
    let Some(l) = (unsafe { loc.as_ref() }) else { return Ok(None) };
    match l.kind {
        CNT_LOCATION_NONE => Ok(None),
        CNT_LOCATION_INLINE => {
            if l.inline_len > continuum_core::MAX_INLINE_BYTES {
                return Err(invalid("inline payloads are at most 64 KiB"));
            }
            if l.inline_bytes.is_null() && l.inline_len > 0 {
                return Err(invalid("inline_bytes is null"));
            }
            let bytes = if l.inline_len == 0 {
                Vec::new()
            } else {
                // SAFETY: non-null and valid for inline_len bytes, as the caller promises.
                unsafe { std::slice::from_raw_parts(l.inline_bytes, l.inline_len) }.to_vec()
            };
            Ok(Some(StorageLocator::Inline { bytes }))
        }
        CNT_LOCATION_SHM => {
            if !l.arena.contains(&0) {
                return Err(invalid("arena isn't NUL-terminated"));
            }
            // SAFETY: the array holds a NUL, checked above.
            let arena =
                unsafe { CStr::from_ptr(l.arena.as_ptr()) }.to_str().map_err(|_| invalid("arena isn't UTF-8"))?;
            Ok(Some(StorageLocator::Shm {
                arena: arena.to_owned(),
                arena_generation: l.arena_generation,
                slab_class: l.slab_class,
                slot: l.slot,
                slot_generation: l.slot_generation,
                offset: l.offset,
                length: l.length,
            }))
        }
        _ => Err(invalid("location kind must be CNT_LOCATION_NONE, _INLINE, or _SHM")),
    }
}

/// A header from bytes the caller holds.
///
/// # Safety
///
/// `bytes` is valid for `len` bytes.
unsafe fn header(bytes: *const u8, len: usize) -> Result<HeaderBuf, cnt_status> {
    if bytes.is_null() {
        return Err(invalid("header is null"));
    }
    // SAFETY: as the caller promises.
    let v = unsafe { std::slice::from_raw_parts(bytes, len) }.to_vec();
    HeaderBuf::new(v).map_err(|e| invalid(&format!("header: {e}")))
}

/// Tells the broker a payload is written: its header (a PayloadHeader FlatBuffer, borrowed) and where its bytes are
/// (null for none). One way: a refusal arrives later from `cnt_next_event` as `CNT_EVENT_INGEST_REJECTED`.
///
/// # Safety
///
/// `client` is a live client; `header_bytes` is valid for `header_len` bytes; `at` is null or a valid location.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_ingest(
    client: *mut cnt_client,
    header_bytes: *const u8,
    header_len: usize,
    at: *const cnt_location,
) -> cnt_status {
    guard(|| {
        // SAFETY: as the caller promises.
        let Some(c) = (unsafe { client.as_ref() }) else { return invalid("client is null") };
        // SAFETY: as the caller promises.
        let h = match unsafe { header(header_bytes, header_len) } {
            Ok(h) => h,
            Err(s) => return s,
        };
        // SAFETY: as the caller promises.
        let at = match unsafe { locator(at) } {
            Ok(at) => at,
            Err(s) => return s,
        };
        match c.client.ingest(&h, at.as_ref()) {
            Ok(()) => cnt_status::CNT_OK,
            Err(e) => from_client(&e),
        }
    })
}

/// Sets the voice turn (the schema's VoiceTurnState: 0 Idle, 1 Listening, 2 Thinking, 3 Speaking). For the voice
/// service. One way.
///
/// # Safety
///
/// `client` is a live client.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_set_system_state(client: *mut cnt_client, voice_turn: u8) -> cnt_status {
    guard(|| {
        // SAFETY: as the caller promises.
        let Some(c) = (unsafe { client.as_ref() }) else { return invalid("client is null") };
        let v = fb::VoiceTurnState(voice_turn);
        if v.variant_name().is_none() {
            return invalid("voice_turn isn't a VoiceTurnState value");
        }
        match c.client.set_system_state(v) {
            Ok(()) => cnt_status::CNT_OK,
            Err(e) => from_client(&e),
        }
    })
}

/// A stage's id: the 32-bit FNV-1a hash of its name (0 for a null or non-UTF-8 name, which no caller should pass).
///
/// # Safety
///
/// `name` is null or a valid NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_stage_id(name: *const c_char) -> u32 {
    // SAFETY: as the caller promises.
    unsafe { str_arg(name) }.map_or(0, |n| continuum_client::stage_id(n).0)
}

// ---- Registering a worker ----

/// An input a capability needs.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cnt_requirement {
    /// The input stream's name.
    pub stream: *const c_char,
    /// Nanoseconds before the output's time the input must cover.
    pub before_ns: u64,
    /// Nanoseconds after.
    pub after_ns: u64,
}

/// What a worker can make and what it needs. Enums are the schema's values.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cnt_capability {
    /// The output stream's name (registered beforehand).
    pub produces: *const c_char,
    /// Its inputs (`n_inputs` of them; null if none). Named `inputs` because `requires` is a C++20 keyword.
    pub inputs: *const cnt_requirement,
    /// How many inputs.
    pub n_inputs: usize,
    /// ComputeDomain: 0 CPU, 1 GPU, 2 ANE, 3 MediaEngine.
    pub compute_domain: u8,
    /// RAM with the model loaded, MB.
    pub ram_mb: u32,
    /// VRAM, MB.
    pub vram_mb: u32,
    /// A first guess at milliseconds per item.
    pub est_ms_per_item: f32,
    /// Model load time, ms.
    pub cold_start_ms: u32,
    /// Most items per job.
    pub max_batch: u32,
    /// Nonzero if a lease may be taken back mid-job.
    pub preemptible: u8,
    /// CapabilityRole: 0 Produce, 1 Hydrate.
    pub role: u8,
    /// The largest payload a job writes, bytes.
    pub output_bytes: u64,
    /// Payloads per job.
    pub outputs_per_job: u32,
    /// Nonzero if `expected_confidence` is set.
    pub has_expected_confidence: u8,
    /// The model's typical confidence.
    pub expected_confidence: f32,
}

/// The broker's answer for one capability, in the order they were given.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct cnt_capability_result {
    /// Nonzero if accepted.
    pub accepted: u8,
    /// The id to quote with (when accepted).
    pub id: u32,
    /// Nonzero if this machine's profile can run it.
    pub runnable: u8,
}

/// Registers this session as a worker. `results` (room for `n_capabilities`) gets each capability's answer in
/// order. Refusal reasons go into `reasons` (NUL-terminated, one per line, cut to `reasons_len`; null to skip).
/// `device_index` is -1 for none; `keeps_copies` is nonzero for a worker that keeps its own copies of payloads (it
/// then gets each purge's `PurgeNotice` and must acknowledge it).
///
/// # Safety
///
/// `client` is live; `worker_id` and `version_hash` (null for none) are valid strings; `capabilities` holds
/// `n_capabilities` valid entries whose strings and arrays are valid; `results` has room for `n_capabilities`;
/// `reasons` is null or writable for `reasons_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_register_worker(
    client: *mut cnt_client,
    worker_id: *const c_char,
    version_hash: *const c_char,
    device_index: i16,
    keeps_copies: u8,
    capabilities: *const cnt_capability,
    n_capabilities: usize,
    results: *mut cnt_capability_result,
    reasons: *mut c_char,
    reasons_len: usize,
) -> cnt_status {
    guard(|| {
        // SAFETY: as the caller promises.
        let Some(c) = (unsafe { client.as_ref() }) else { return invalid("client is null") };
        // SAFETY: as the caller promises.
        let Some(worker_id) = (unsafe { str_arg(worker_id) }) else { return invalid("worker_id is null") };
        // SAFETY: as the caller promises.
        let version_hash = unsafe { str_arg(version_hash) }.unwrap_or("");
        if n_capabilities > 0 && (capabilities.is_null() || results.is_null()) {
            return invalid("capabilities or results is null");
        }
        let caps = if n_capabilities == 0 {
            &[][..]
        } else {
            // SAFETY: non-null and valid for n_capabilities entries, as the caller promises.
            unsafe { std::slice::from_raw_parts(capabilities, n_capabilities) }
        };
        let mut specs = Vec::with_capacity(caps.len());
        for cap in caps {
            // SAFETY: the capability's strings are valid, as the caller promises.
            let Some(produces) = (unsafe { str_arg(cap.produces) }) else { return invalid("produces is null") };
            let domain = fb::ComputeDomain(cap.compute_domain);
            let role = fb::CapabilityRole(cap.role);
            if domain.variant_name().is_none() || role.variant_name().is_none() {
                return invalid("compute_domain or role isn't one of the schema's values");
            }
            let mut spec = CapabilitySpec::new(produces, domain);
            if cap.n_inputs > 0 {
                if cap.inputs.is_null() {
                    return invalid("inputs is null");
                }
                // SAFETY: non-null and valid for n_inputs entries, as the caller promises.
                for r in unsafe { std::slice::from_raw_parts(cap.inputs, cap.n_inputs) } {
                    // SAFETY: as the caller promises.
                    let Some(stream) = (unsafe { str_arg(r.stream) }) else {
                        return invalid("a requirement's stream is null");
                    };
                    spec.requires.push(Requirement {
                        stream: stream.to_owned(),
                        before_ns: r.before_ns,
                        after_ns: r.after_ns,
                    });
                }
            }
            spec.ram_mb = cap.ram_mb;
            spec.vram_mb = cap.vram_mb;
            spec.est_ms_per_item = cap.est_ms_per_item;
            spec.cold_start_ms = cap.cold_start_ms;
            spec.max_batch = cap.max_batch;
            spec.preemptible = cap.preemptible != 0;
            spec.role = role;
            spec.output_bytes = cap.output_bytes;
            spec.outputs_per_job = cap.outputs_per_job;
            spec.expected_confidence = (cap.has_expected_confidence != 0).then_some(cap.expected_confidence);
            specs.push(spec);
        }
        let device_index = match device_index {
            -1 => None,
            i => match u8::try_from(i) {
                Ok(i) => Some(i),
                Err(_) => return invalid("device_index must be -1 (none) or 0 to 255"),
            },
        };
        let spec = WorkerSpec {
            worker_id: worker_id.to_owned(),
            version_hash: version_hash.to_owned(),
            capabilities: specs,
            device_index,
            keeps_copies: keeps_copies != 0,
        };
        match c.runtime.block_on(c.client.register_worker(&spec)) {
            Ok(r) => {
                for i in 0..n_capabilities {
                    let index = u32::try_from(i).unwrap_or(u32::MAX);
                    let answer = r.accepted.iter().find(|a| a.index == index);
                    let out = cnt_capability_result {
                        accepted: u8::from(answer.is_some()),
                        id: answer.map_or(0, |a| a.id),
                        runnable: u8::from(answer.is_some_and(|a| a.runnable)),
                    };
                    // SAFETY: results has room for n_capabilities, as the caller promises.
                    unsafe { *results.add(i) = out };
                }
                if !reasons.is_null() && reasons_len > 0 {
                    let joined = r.refused_reasons.join("\n");
                    let n = joined.len().min(reasons_len - 1);
                    // SAFETY: reasons is writable for reasons_len bytes, as the caller promises; n < reasons_len.
                    unsafe {
                        ptr::copy_nonoverlapping(joined.as_ptr().cast::<c_char>(), reasons, n);
                        *reasons.add(n) = 0;
                    }
                }
                cnt_status::CNT_OK
            }
            Err(e) => from_client(&e),
        }
    })
}

// ---- Leases ----

/// A granted lease. Opaque: made by `cnt_quote`, ended (and freed) by `cnt_work_lease_commit`,
/// `cnt_work_lease_abort`, or `cnt_work_lease_release`. After its client is freed its calls return `CNT_CLOSED`; release
/// it then to free it.
pub struct cnt_work_lease {
    /// None only while a commit is in flight.
    lease: Option<WorkLease>,
    runtime: tokio::runtime::Handle,
}

/// A lease to ask for (`QuoteRequest`). Enums are the schema's values.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cnt_quote_spec {
    /// The capability's id, from `cnt_register_worker`.
    pub capability_id: u32,
    /// The stage's name.
    pub stage: *const c_char,
    /// PriorityBand: 0 P0Critical, 1 P1Realtime, 2 P2Nearline, 3 P3Background.
    pub band: u8,
    /// ComputeDomain.
    pub compute_domain: u8,
    /// Output bytes wanted.
    pub output_bytes: u64,
    /// Expected duration, ms.
    pub est_duration_ms: u32,
    /// Parents to pin: `n_pin_parents` ids of 16 bytes each, back to back (null if none).
    pub pin_parents: *const u8,
    /// How many.
    pub n_pin_parents: usize,
    /// The time covered, UTC ns, half open.
    pub range_start_ns: u64,
    /// Its end.
    pub range_end_ns: u64,
}

/// Asks for a lease. On `CNT_OK`, `*out` is a lease the caller owns. If the broker says not now, returns
/// `CNT_DENIED` with `*retry_after_ms` (if non-null) set to how long to wait.
///
/// # Safety
///
/// `client` is live; `spec` and its strings and arrays are valid; `out` is writable; `retry_after_ms` is null or
/// writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_quote(
    client: *mut cnt_client,
    spec: *const cnt_quote_spec,
    out: *mut *mut cnt_work_lease,
    retry_after_ms: *mut u32,
) -> cnt_status {
    guard(|| {
        if out.is_null() {
            return invalid("out is null");
        }
        // SAFETY: writable, as the caller promises.
        unsafe { *out = ptr::null_mut() };
        // SAFETY: as the caller promises.
        let (Some(c), Some(q)) = (unsafe { client.as_ref() }, unsafe { spec.as_ref() }) else {
            return invalid("client or spec is null");
        };
        // SAFETY: as the caller promises.
        let Some(stage) = (unsafe { str_arg(q.stage) }) else { return invalid("stage is null") };
        let (band, domain) = (fb::PriorityBand(q.band), fb::ComputeDomain(q.compute_domain));
        if band.variant_name().is_none() || domain.variant_name().is_none() {
            return invalid("band or compute_domain isn't one of the schema's values");
        }
        let parents = if q.n_pin_parents == 0 {
            Vec::new()
        } else if q.pin_parents.is_null() {
            return invalid("pin_parents is null");
        } else {
            let Some(len) = q.n_pin_parents.checked_mul(16) else { return invalid("n_pin_parents is too large") };
            // SAFETY: valid for 16 * n_pin_parents bytes, as the caller promises.
            let raw = unsafe { std::slice::from_raw_parts(q.pin_parents, len) };
            raw.chunks_exact(16).map(|c| Uuid(c.try_into().expect("16-byte chunks"))).collect()
        };
        let quote = QuoteSpec {
            capability_id: q.capability_id,
            stage: stage.to_owned(),
            band,
            compute_domain: domain,
            output_bytes: q.output_bytes,
            est_duration_ms: q.est_duration_ms,
            pin_parents: parents,
            range: (q.range_start_ns, q.range_end_ns),
        };
        match c.runtime.block_on(c.client.quote(&quote)) {
            Ok(Quote::Granted(lease)) => {
                let handle = Box::new(cnt_work_lease { lease: Some(lease), runtime: c.runtime.handle().clone() });
                // SAFETY: writable, checked above.
                unsafe { *out = Box::into_raw(handle) };
                cnt_status::CNT_OK
            }
            Ok(Quote::Denied { retry_after_ms: wait }) => {
                if !retry_after_ms.is_null() {
                    // SAFETY: non-null and writable, as the caller promises.
                    unsafe { *retry_after_ms = wait };
                }
                set_error(cnt_status::CNT_DENIED, &format!("not granted now; retry after {wait} ms"), 0)
            }
            Err(e) => from_client(&e),
        }
    })
}

/// A handle for `lease`, which the caller owns.
pub(crate) fn work_lease(lease: WorkLease, runtime: tokio::runtime::Handle) -> *mut cnt_work_lease {
    Box::into_raw(Box::new(cnt_work_lease { lease: Some(lease), runtime }))
}

/// The lease's id (what a `CNT_EVENT_LEASE_PREEMPTED` names). 0 for null.
///
/// # Safety
///
/// `lease` is null or live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_work_lease_id(lease: *const cnt_work_lease) -> u64 {
    // SAFETY: as the caller promises.
    unsafe { lease.as_ref() }.and_then(|l| l.lease.as_ref()).map_or(0, WorkLease::id)
}

/// When the lease expires, UTC ns. 0 for null.
///
/// # Safety
///
/// `lease` is null or live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_work_lease_expires_at_ns(lease: *const cnt_work_lease) -> u64 {
    // SAFETY: as the caller promises.
    unsafe { lease.as_ref() }.and_then(|l| l.lease.as_ref()).map_or(0, WorkLease::expires_at_ns)
}

/// Renews the lease by `extend_ms` (the broker caps it) and writes the new expiry to `*expires_at_ns` (if non-null).
///
/// # Safety
///
/// `lease` is live and its client too; `expires_at_ns` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_work_lease_renew(
    lease: *mut cnt_work_lease,
    extend_ms: u32,
    expires_at_ns: *mut u64,
) -> cnt_status {
    guard(|| {
        // SAFETY: as the caller promises.
        let Some(l) = (unsafe { lease.as_ref() }) else { return invalid("lease is null") };
        let Some(work) = l.lease.as_ref() else { return invalid("the lease already ended") };
        match l.runtime.block_on(work.renew(Duration::from_millis(extend_ms.into()))) {
            Ok(e) => {
                if !expires_at_ns.is_null() {
                    // SAFETY: writable, as the caller promises.
                    unsafe { *expires_at_ns = e };
                }
                cnt_status::CNT_OK
            }
            Err(e) => from_client(&e),
        }
    })
}

/// Renews the lease in the background every `every_ms` (at least 10) until it ends. It doesn't stop on a preemption
/// (the broker still renews a lease it's taking back): watch `cnt_next_event` for a `CNT_EVENT_LEASE_PREEMPTED`
/// naming this lease and end it then.
///
/// # Safety
///
/// `lease` is live and used by no other thread during the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_work_lease_keep_alive(
    lease: *mut cnt_work_lease,
    every_ms: u32,
    extend_ms: u32,
) -> cnt_status {
    guard(|| {
        // SAFETY: as the caller promises.
        let Some(l) = (unsafe { lease.as_mut() }) else { return invalid("lease is null") };
        if every_ms < 10 {
            return invalid("every_ms must be at least 10");
        }
        let Some(work) = l.lease.as_mut() else { return invalid("the lease already ended") };
        work.keep_alive(Duration::from_millis(every_ms.into()), Duration::from_millis(extend_ms.into()));
        cnt_status::CNT_OK
    })
}

/// One output of a job: its header (a PayloadHeader FlatBuffer) and where its bytes are.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cnt_output {
    /// The header's bytes (borrowed for the call).
    pub header: *const u8,
    /// How many.
    pub header_len: usize,
    /// Where the payload's bytes are (null for none).
    pub at: *const cnt_location,
}

/// The coverage a commit finished.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cnt_coverage {
    /// The stream the work read.
    pub source_stream_id: u32,
    /// The stage's name.
    pub stage: *const c_char,
    /// The time covered, UTC ns, half open.
    pub range_start_ns: u64,
    /// Its end.
    pub range_end_ns: u64,
}

/// A job's measured cost.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cnt_cost {
    /// Wall time, us.
    pub wall_time_us: u64,
    /// Queued time, us.
    pub queue_wait_us: u64,
    /// Peak RAM, bytes.
    pub ram_bytes: u64,
    /// Peak VRAM, bytes.
    pub vram_bytes: u64,
    /// Like "CUDA:0" (null for none).
    pub device: *const c_char,
    /// Items processed.
    pub items: u64,
}

/// Takes ownership of a lease handle back from C.
///
/// # Safety
///
/// `lease` is a live handle from `cnt_quote`, not used again.
unsafe fn take(lease: *mut cnt_work_lease) -> Box<cnt_work_lease> {
    // SAFETY: as the caller promises.
    unsafe { Box::from_raw(lease) }
}

/// Commits the job: `outputs` (`n_outputs` of them), the coverage it finished (null for none), and its cost (null for
/// none). Returns once the broker has the outputs durably.
///
/// Who owns the lease afterward:
/// - `CNT_OK`: the lease is over and the handle freed.
/// - `CNT_INVALID_ARGUMENT`: nothing was sent; the handle is still the caller's.
/// - `CNT_BROKER` with broker code Unavailable (5): the broker was busy; the handle is still the caller's, to commit
///   again or release.
/// - `CNT_PANIC` (a bug): the handle is still the caller's, though its lease may be gone; release it.
/// - Anything else: the lease was given back and the handle freed.
///
/// # Safety
///
/// `lease` is a live handle from `cnt_quote`, used by no other thread during the call; every pointer is null where
/// allowed or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_work_lease_commit(
    lease: *mut cnt_work_lease,
    outputs: *const cnt_output,
    n_outputs: usize,
    coverage: *const cnt_coverage,
    cost: *const cnt_cost,
) -> cnt_status {
    guard(|| {
        if lease.is_null() {
            return invalid("lease is null");
        }
        if n_outputs > 0 && outputs.is_null() {
            return invalid("outputs is null");
        }
        let mut outs = Vec::with_capacity(n_outputs);
        if n_outputs > 0 {
            // SAFETY: valid for n_outputs entries, as the caller promises.
            for o in unsafe { std::slice::from_raw_parts(outputs, n_outputs) } {
                // SAFETY: as the caller promises.
                let h = match unsafe { header(o.header, o.header_len) } {
                    Ok(h) => h,
                    Err(s) => return s,
                };
                // SAFETY: as the caller promises.
                let at = match unsafe { locator(o.at) } {
                    Ok(at) => at,
                    Err(s) => return s,
                };
                outs.push((h, at));
            }
        }
        // SAFETY: as the caller promises.
        let coverage = match unsafe { coverage.as_ref() } {
            None => None,
            Some(c) => {
                // SAFETY: as the caller promises.
                let Some(stage) = (unsafe { str_arg(c.stage) }) else { return invalid("coverage stage is null") };
                Some(CoverageDone {
                    source_stream_id: c.source_stream_id,
                    stage: stage.to_owned(),
                    range: (c.range_start_ns, c.range_end_ns),
                })
            }
        };
        // SAFETY: as the caller promises.
        let cost = unsafe { cost.as_ref() }.map(|c| Cost {
            wall_time_us: c.wall_time_us,
            queue_wait_us: c.queue_wait_us,
            ram_bytes: c.ram_bytes,
            vram_bytes: c.vram_bytes,
            // SAFETY: as the caller promises.
            device: unsafe { str_arg(c.device) }.unwrap_or("").to_owned(),
            items: c.items,
        });
        // Arguments checked. The handle stays allocated until the broker answers: a busy refusal hands the lease
        // back into it for a retry.
        // SAFETY: a live handle, as the caller promises.
        let handle = unsafe { &mut *lease };
        let Some(l) = handle.lease.take() else { return invalid("the lease already ended") };
        let result = handle.runtime.block_on(l.commit(&outs, coverage.as_ref(), cost.as_ref()));
        match result {
            Ok(()) => {
                // SAFETY: the caller doesn't use the handle again.
                drop(unsafe { take(lease) });
                cnt_status::CNT_OK
            }
            Err(e) => {
                let status = from_client(&e.error);
                match e.lease {
                    Some(back) => handle.lease = Some(back),
                    // SAFETY: the caller doesn't use the handle again.
                    None => drop(unsafe { take(lease) }),
                }
                status
            }
        }
    })
}

/// Gives up the job with a reason. Returns once the broker has done so. The lease is over and freed either way.
///
/// # Safety
///
/// `lease` is a live handle from `cnt_quote`, not used again; `reason` is null or a valid string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_work_lease_abort(lease: *mut cnt_work_lease, reason: *const c_char) -> cnt_status {
    guard(|| {
        if lease.is_null() {
            return invalid("lease is null");
        }
        // SAFETY: as the caller promises.
        let cnt_work_lease { lease, runtime } = *unsafe { take(lease) };
        let Some(lease) = lease else { return invalid("the lease already ended") };
        // SAFETY: as the caller promises.
        let reason = unsafe { str_arg(reason) }.unwrap_or("aborted");
        match runtime.block_on(lease.abort(reason)) {
            Ok(()) => cnt_status::CNT_OK,
            Err(e) => from_client(&e),
        }
    })
}

/// Gives the lease back without committing (after a preemption, or early) and frees it. One way. Null is a no-op.
///
/// # Safety
///
/// `lease` is null or a live handle from `cnt_quote`, not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_work_lease_release(lease: *mut cnt_work_lease) {
    let _ = guard(|| {
        if !lease.is_null() {
            // SAFETY: as the caller promises.
            let l = unsafe { take(lease) };
            let _enter = l.runtime.enter(); // dropping the keep-alive's handle may touch the runtime
            if let Some(work) = l.lease {
                work.release();
            }
        }
        cnt_status::CNT_OK
    });
}

// ---- Events ----

/// `cnt_event.type`: a message type this ABI doesn't decode yet (`payload_type` says which): a purge notice, or
/// anything newer.
pub const CNT_EVENT_OTHER: u8 = 0;
/// `cnt_event.type`: a lease is being taken back.
pub const CNT_EVENT_LEASE_PREEMPTED: u8 = 1;
/// `cnt_event.type`: the voice turn changed.
pub const CNT_EVENT_SYSTEM_STATE_CHANGED: u8 = 2;
/// `cnt_event.type`: an ingest notice was refused.
pub const CNT_EVENT_INGEST_REJECTED: u8 = 3;
/// `cnt_event.type`: the broker refused something this session sent one way; `reason` has its message.
pub const CNT_EVENT_ERROR: u8 = 4;
/// `cnt_event.type`: a job for this worker; `job` is the caller's (`cnt_job_take_lease`, then `cnt_job_free`).
pub const CNT_EVENT_DISPATCH_JOB: u8 = 5;
/// `cnt_event.type`: outputs of a committed promise no `cnt_promise` holds: `promise_id`, `items` (the caller's, to
/// `cnt_items_free`), and `final_part`.
pub const CNT_EVENT_PROMISE_FULFILLED: u8 = 6;
/// `cnt_event.type`: a committed promise no `cnt_promise` holds failed: `promise_id` and `reason`.
pub const CNT_EVENT_PROMISE_FAILED: u8 = 7;

/// Something the broker pushed. Which fields are set depends on `type`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cnt_event {
    /// One of the `CNT_EVENT_*` values.
    pub r#type: u8,
    /// The schema's ControlPayload value of the message.
    pub payload_type: u8,
    /// Preempted: the lease.
    pub lease_id: u64,
    /// Preempted: the band that took it.
    pub by_band: u8,
    /// Preempted: nonzero if the broker took it back on its own ("revoked:").
    pub revoked: u8,
    /// Preempted or rejected: why (NUL-terminated, cut to fit).
    pub reason: [c_char; CNT_REASON_MAX],
    /// State changed: the new VoiceTurnState.
    pub voice_turn: u8,
    /// State changed: when, UTC ns.
    pub changed_at_ns: u64,
    /// Rejected: the stream.
    pub stream_id: u32,
    /// Rejected: nonzero if `payload_id` is set.
    pub has_payload_id: u8,
    /// Rejected: the payload's id.
    pub payload_id: [u8; 16],
    /// Dispatch: the job, which the caller owns (null otherwise).
    pub job: *mut crate::data::cnt_job,
    /// Promise fulfilled or failed: the promise.
    pub promise_id: [u8; 16],
    /// Promise fulfilled: the outputs, which the caller owns (null otherwise).
    pub items: *mut crate::data::cnt_items,
    /// Promise fulfilled: nonzero for the last part.
    pub final_part: u8,
}

pub(crate) fn copy_reason(s: &str, out: &mut [c_char; CNT_REASON_MAX]) {
    out.fill(0);
    for (d, &b) in out.iter_mut().zip(s.as_bytes().iter().take(CNT_REASON_MAX - 1)) {
        *d = b as c_char;
    }
}

/// Waits up to `timeout_ms` for the next message the broker pushes and writes it to `*out`. `CNT_TIMEOUT` if none
/// came, `CNT_CLOSED` once the connection is closed.
///
/// # Safety
///
/// `client` is live and used by no other thread during the call; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_next_event(client: *mut cnt_client, timeout_ms: u32, out: *mut cnt_event) -> cnt_status {
    guard(|| {
        // SAFETY: as the caller promises.
        let Some(c) = (unsafe { client.as_ref() }) else { return invalid("client is null") };
        if out.is_null() {
            return invalid("out is null");
        }
        let wait = Duration::from_millis(timeout_ms.into());
        let got = c.runtime.block_on(async { tokio::time::timeout(wait, c.client.next_event()).await });
        let message = match got {
            Err(_) => return set_error(cnt_status::CNT_TIMEOUT, "no event within the timeout", 0),
            Ok(None) => return set_error(cnt_status::CNT_CLOSED, "the connection is closed", 0),
            Ok(Some(m)) => m,
        };
        let mut e = cnt_event {
            r#type: CNT_EVENT_OTHER,
            payload_type: message.kind().0,
            lease_id: 0,
            by_band: 0,
            revoked: 0,
            reason: [0; CNT_REASON_MAX],
            voice_turn: 0,
            changed_at_ns: 0,
            stream_id: 0,
            has_payload_id: 0,
            payload_id: [0; 16],
            job: ptr::null_mut(),
            promise_id: [0; 16],
            items: ptr::null_mut(),
            final_part: 0,
        };
        let event = message.event();
        let revoked = event.is_revocation();
        match event {
            BrokerEvent::LeasePreempted { lease_id, reason, by_band } => {
                e.r#type = CNT_EVENT_LEASE_PREEMPTED;
                e.lease_id = lease_id;
                e.by_band = by_band.0;
                e.revoked = u8::from(revoked);
                copy_reason(&reason, &mut e.reason);
            }
            BrokerEvent::SystemStateChanged { voice_turn, changed_at_ns } => {
                e.r#type = CNT_EVENT_SYSTEM_STATE_CHANGED;
                e.voice_turn = voice_turn.0;
                e.changed_at_ns = changed_at_ns;
            }
            BrokerEvent::IngestRejected { stream_id, payload_id, reason } => {
                e.r#type = CNT_EVENT_INGEST_REJECTED;
                e.stream_id = stream_id;
                if let Some(p) = payload_id {
                    e.has_payload_id = 1;
                    e.payload_id = p.0;
                }
                copy_reason(&reason, &mut e.reason);
            }
            BrokerEvent::Error { message, .. } => {
                e.r#type = CNT_EVENT_ERROR;
                copy_reason(&message, &mut e.reason);
            }
            BrokerEvent::DispatchJob(job) => {
                e.r#type = CNT_EVENT_DISPATCH_JOB;
                e.job = crate::data::job_handle(*job);
            }
            BrokerEvent::Fulfilled { promise_id, items, final_part } => {
                e.r#type = CNT_EVENT_PROMISE_FULFILLED;
                e.promise_id = promise_id.0;
                e.items = crate::data::items_handle(items);
                e.final_part = u8::from(final_part);
            }
            BrokerEvent::PromiseFailed { promise_id, reason, .. } => {
                e.r#type = CNT_EVENT_PROMISE_FAILED;
                e.promise_id = promise_id.0;
                copy_reason(&reason, &mut e.reason);
            }
            // Not decoded for C yet: they arrive as CNT_EVENT_OTHER with their payload_type, so a C client sees them
            // and can ignore them.
            BrokerEvent::PurgeNotice { .. } | BrokerEvent::Undecodable { .. } => {}
            BrokerEvent::Other(_) => {}
        }
        // SAFETY: writable, as the caller promises.
        unsafe { *out = e };
        cnt_status::CNT_OK
    })
}
