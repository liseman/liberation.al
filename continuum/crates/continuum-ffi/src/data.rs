//! The data side of the C ABI: claiming and publishing Tier 0 slots, queries and the consumer's promises (quote,
//! commit, cancel, the outputs as they come), subscriptions, and the jobs the broker dispatches. The same rules as
//! the rest of the ABI (see the crate docs): integers rather than enums for anything C passes in, explicit
//! ownership (every handle here has its `_free`), and a status code from every call that can fail.
//!
//! Item lists (`cnt_items`) are what queries, promise parts, subscription events, and job inputs hand back: a list of
//! payloads, each a header and where its bytes are. An inline payload's bytes and a header point into the list, so
//! they're valid until the list is freed.

use std::ffi::c_char;
use std::ptr;
use std::time::Duration;

use continuum_client::{
    DispatchJob, Item, Promise, PromiseQuote, QuerySpec, Start, SubEvent, SubscribeSpec, Subscription,
};
use continuum_core::schema::continuum as fb;
use continuum_core::{StorageLocator, Uuid};
use continuum_shm::WriteSlot;

use crate::worker::{
    CNT_LOCATION_INLINE, CNT_LOCATION_NONE, CNT_LOCATION_OTHER, CNT_LOCATION_SHM, CNT_REASON_MAX, cnt_location,
    cnt_work_lease, copy_reason, work_lease,
};
use crate::{
    CNT_NAME_MAX, cnt_client, cnt_status, copy_name, from_client, from_shm, guard, invalid, set_error, str_arg,
};

fn not_found(message: &str) -> cnt_status {
    set_error(cnt_status::CNT_NOT_FOUND, message, 0)
}

/// Waits up to `timeout_ms` for `f` on `runtime`: `None` when the time ran out.
fn within<T>(runtime: &tokio::runtime::Handle, timeout_ms: u32, f: impl Future<Output = T>) -> Option<T> {
    runtime.block_on(async { tokio::time::timeout(Duration::from_millis(timeout_ms.into()), f).await.ok() })
}

/// A 16-byte id the caller passes.
///
/// # Safety
///
/// `id` is null or valid for 16 bytes.
unsafe fn uuid_arg(id: *const u8) -> Option<Uuid> {
    if id.is_null() {
        return None;
    }
    // SAFETY: valid for 16 bytes, as the caller promises.
    Some(Uuid(unsafe { std::slice::from_raw_parts(id, 16) }.try_into().expect("16 bytes")))
}

// ---- Slots ----

/// A Tier 0 slot claimed for writing: fill `cnt_write_slot_data` (up to `cnt_write_slot_capacity` bytes), then
/// publish it. Opaque: made by `cnt_claim_slot`, ended by `cnt_write_slot_publish` or `cnt_write_slot_abandon`.
pub struct cnt_write_slot {
    slot: WriteSlot,
    arena: String,
    arena_generation: u32,
}

/// Claims a free slot of size class `slab_class` in arena `arena_index` (in Welcome order) for writing, tagged with
/// `owner` (a stream id, say; 0 for none), and writes it to `*out`, which the caller owns until it publishes or
/// abandons it. `CNT_ARENA` when the class has no free slot.
///
/// # Safety
///
/// `client` is a live client; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_claim_slot(
    client: *const cnt_client,
    arena_index: usize,
    slab_class: u16,
    owner: u64,
    out: *mut *mut cnt_write_slot,
) -> cnt_status {
    guard(|| {
        if out.is_null() {
            return invalid("out is null");
        }
        // SAFETY: writable, as the caller promises.
        unsafe { *out = ptr::null_mut() };
        // SAFETY: as the caller promises.
        let Some(c) = (unsafe { client.as_ref() }) else { return invalid("client is null") };
        let Some(a) = c.client.arenas().get(arena_index) else { return not_found(&format!("no arena {arena_index}")) };
        match a.arena.claim(slab_class, owner) {
            Ok(slot) => {
                let handle = cnt_write_slot { slot, arena: a.info.name.clone(), arena_generation: a.info.generation };
                // SAFETY: writable, checked above.
                unsafe { *out = Box::into_raw(Box::new(handle)) };
                cnt_status::CNT_OK
            }
            Err(e) => from_shm(&e),
        }
    })
}

/// The claimed slot's bytes, to fill before publishing (16 KB aligned, `cnt_write_slot_capacity` long). Null for
/// null.
///
/// # Safety
///
/// `slot` is null or a live claimed slot, used by no other thread during the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_write_slot_data(slot: *mut cnt_write_slot) -> *mut u8 {
    // SAFETY: as the caller promises.
    unsafe { slot.as_mut() }.map_or(ptr::null_mut(), |s| s.slot.as_mut_slice().as_mut_ptr())
}

/// How many bytes the claimed slot holds. 0 for null.
///
/// # Safety
///
/// `slot` is null or a live claimed slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_write_slot_capacity(slot: *const cnt_write_slot) -> usize {
    // SAFETY: as the caller promises.
    unsafe { slot.as_ref() }.map_or(0, |s| s.slot.capacity())
}

/// Publishes the slot's first `len` bytes (readers can lease it from now on) and writes where they are to `*at`:
/// a `CNT_LOCATION_SHM` location for `cnt_ingest` or a commit's output. The handle is freed either way.
/// `CNT_STALE` if the broker took the slot back meanwhile (it thought this producer crashed).
///
/// # Safety
///
/// `slot` is a live claimed slot, not used again; `at` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_write_slot_publish(
    slot: *mut cnt_write_slot,
    len: usize,
    at: *mut cnt_location,
) -> cnt_status {
    guard(|| {
        if slot.is_null() || at.is_null() {
            return invalid("slot or at is null");
        }
        // SAFETY: made by cnt_claim_slot and not used again, as the caller promises.
        let s = unsafe { Box::from_raw(slot) };
        let cnt_write_slot { slot, arena, arena_generation } = *s;
        if len > slot.capacity() {
            return invalid(&format!("{len} bytes don't fit a {}-byte slot", slot.capacity()));
        }
        let offset = slot.offset() as u64;
        match slot.publish(len) {
            Ok(handle) => {
                let mut loc = empty_location();
                loc.kind = CNT_LOCATION_SHM;
                copy_name(&arena, &mut loc.arena);
                loc.arena_generation = arena_generation;
                loc.slab_class = handle.class;
                loc.slot = handle.index;
                loc.slot_generation = handle.generation;
                loc.offset = offset;
                loc.length = len as u64;
                // SAFETY: writable, as the caller promises.
                unsafe { *at = loc };
                cnt_status::CNT_OK
            }
            Err(e) => from_shm(&e),
        }
    })
}

/// Gives a claimed slot back unpublished and frees the handle. Null is a no-op.
///
/// # Safety
///
/// `slot` is null or a live claimed slot, not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_write_slot_abandon(slot: *mut cnt_write_slot) {
    let _ = guard(|| {
        if !slot.is_null() {
            // SAFETY: made by cnt_claim_slot and not used again, as the caller promises.
            drop(unsafe { Box::from_raw(slot) });
        }
        cnt_status::CNT_OK
    });
}

// ---- Item lists ----

/// A list of payloads (headers and where their bytes are). Opaque: from a query, a promise's part, a subscription
/// event, a job, or a `CNT_EVENT_PROMISE_FULFILLED`; freed by `cnt_items_free`.
pub struct cnt_items {
    items: Vec<Item>,
}

/// One payload of a `cnt_items`. Its pointers are into the list: valid until the list is freed.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cnt_item {
    /// The PayloadHeader FlatBuffer.
    pub header: *const u8,
    /// Its length.
    pub header_len: usize,
    /// Where the bytes are: `CNT_LOCATION_NONE` for a header-only payload, `_INLINE`, `_SHM` (lease it with
    /// `cnt_lease_slot`), or `_OTHER` (in a chunk or a ring, which this ABI doesn't read yet).
    pub at: cnt_location,
}

pub(crate) fn empty_location() -> cnt_location {
    cnt_location {
        kind: CNT_LOCATION_NONE,
        inline_bytes: ptr::null(),
        inline_len: 0,
        arena: [0; CNT_NAME_MAX],
        arena_generation: 0,
        slab_class: 0,
        slot: 0,
        slot_generation: 0,
        offset: 0,
        length: 0,
    }
}

/// Where a payload's bytes are, as C reads it; an inline payload's bytes are borrowed from `at`.
fn location_of(at: Option<&StorageLocator>) -> cnt_location {
    let mut loc = empty_location();
    match at {
        None => {}
        Some(StorageLocator::Inline { bytes }) => {
            loc.kind = CNT_LOCATION_INLINE;
            loc.inline_bytes = bytes.as_ptr();
            loc.inline_len = bytes.len();
        }
        Some(StorageLocator::Shm { arena, arena_generation, slab_class, slot, slot_generation, offset, length }) => {
            loc.kind = CNT_LOCATION_SHM;
            copy_name(arena, &mut loc.arena);
            loc.arena_generation = *arena_generation;
            loc.slab_class = *slab_class;
            loc.slot = *slot;
            loc.slot_generation = *slot_generation;
            loc.offset = *offset;
            loc.length = *length;
        }
        Some(_) => loc.kind = CNT_LOCATION_OTHER,
    }
    loc
}

pub(crate) fn items_handle(items: Vec<Item>) -> *mut cnt_items {
    Box::into_raw(Box::new(cnt_items { items }))
}

/// How many payloads the list holds. 0 for null.
///
/// # Safety
///
/// `items` is null or a live list.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_items_count(items: *const cnt_items) -> usize {
    // SAFETY: as the caller promises.
    unsafe { items.as_ref() }.map_or(0, |i| i.items.len())
}

/// Writes payload `index` to `*out`. `CNT_NOT_FOUND` past the last one.
///
/// # Safety
///
/// `items` is a live list; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_items_at(items: *const cnt_items, index: usize, out: *mut cnt_item) -> cnt_status {
    guard(|| {
        // SAFETY: as the caller promises.
        let Some(list) = (unsafe { items.as_ref() }) else { return invalid("items is null") };
        if out.is_null() {
            return invalid("out is null");
        }
        let Some(item) = list.items.get(index) else { return not_found(&format!("no item {index}")) };
        let header = item.header.as_bytes();
        let got =
            cnt_item { header: header.as_ptr(), header_len: header.len(), at: location_of(item.locator.as_ref()) };
        // SAFETY: writable, as the caller promises.
        unsafe { *out = got };
        cnt_status::CNT_OK
    })
}

/// Frees a list. Null is a no-op.
///
/// # Safety
///
/// `items` is null or a live list the caller owns, not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_items_free(items: *mut cnt_items) {
    if !items.is_null() {
        // SAFETY: made by Box::into_raw here and not freed before, as the caller promises.
        drop(unsafe { Box::from_raw(items) });
    }
}

// ---- Queries and promises ----

/// A query. Enums are the schema's values.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cnt_query_spec {
    /// The stream, by name.
    pub stream: *const c_char,
    /// The time range, UTC ns, half open.
    pub range_start_ns: u64,
    /// Its end.
    pub range_end_ns: u64,
    /// QueryMode: 0 HitOnly, 1 QuoteIfMissing, 2 CommitWithinBudget.
    pub mode: u8,
    /// PriorityBand to quote or run missing work at: 0 P0Critical to 3 P3Background.
    pub band: u8,
    /// A latency budget, ms; 0 for none.
    pub latency_budget_ms: u32,
    /// At most this many items; 0 for the broker's own limit.
    pub max_items: u32,
}

/// What a query found. Opaque: made by `cnt_query`, freed by `cnt_query_result_free`.
pub struct cnt_query_result {
    items: cnt_items,
    missing: Vec<(u64, u64)>,
    complete: bool,
    truncated: bool,
    quote: Option<PromiseQuote>,
}

/// A quote for what a query found missing: commit it with `cnt_commit_promise` before `expires_at_ns`, or let it
/// lapse. The estimates are the broker's pick's (option 0).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cnt_promise_quote {
    /// The promise to commit or cancel.
    pub promise_id: [u8; 16],
    /// PromiseStatus: what it waits on (compute, capture, or both).
    pub status: u8,
    /// How many options it offers (0 is the broker's pick).
    pub n_options: u32,
    /// Option 0's estimated latency, ms.
    pub est_latency_ms: f32,
    /// Option 0's estimated cost.
    pub cost_units: f32,
    /// Nonzero if option 0 runs past the query's latency budget.
    pub exceeds_budget: u8,
    /// Nonzero if work already running covers some of it (committing joins it).
    pub shares_in_flight: u8,
    /// When the quote lapses, UTC ns.
    pub expires_at_ns: u64,
}

/// Asks what `spec.stream` has over the range and writes the answer to `*out`, which the caller owns. In a quoting
/// mode, the answer also carries the quote for what's missing (`cnt_query_quote`) when anything missing can be made.
///
/// # Safety
///
/// `client` is live; `spec` and its string are valid; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_query(
    client: *mut cnt_client,
    spec: *const cnt_query_spec,
    out: *mut *mut cnt_query_result,
) -> cnt_status {
    guard(|| {
        if out.is_null() {
            return invalid("out is null");
        }
        // SAFETY: writable, as the caller promises.
        unsafe { *out = ptr::null_mut() };
        // SAFETY: as the caller promises.
        let (Some(c), Some(s)) = (unsafe { client.as_ref() }, unsafe { spec.as_ref() }) else {
            return invalid("client or spec is null");
        };
        // SAFETY: as the caller promises.
        let Some(stream) = (unsafe { str_arg(s.stream) }) else { return invalid("stream is null") };
        let (mode, band) = (fb::QueryMode(s.mode), fb::PriorityBand(s.band));
        if mode.variant_name().is_none() || band.variant_name().is_none() {
            return invalid("mode or band isn't one of the schema's values");
        }
        let mut q = QuerySpec::hit_only(stream, (s.range_start_ns, s.range_end_ns));
        q.mode = mode;
        q.band = band;
        q.latency_budget_ms = s.latency_budget_ms;
        q.max_items = s.max_items;
        match c.runtime.block_on(c.client.query_quoted(&q)) {
            Ok((r, quote)) => {
                let result = cnt_query_result {
                    items: cnt_items { items: r.items },
                    missing: r.missing,
                    complete: r.complete,
                    truncated: r.truncated,
                    quote,
                };
                // SAFETY: writable, checked above.
                unsafe { *out = Box::into_raw(Box::new(result)) };
                cnt_status::CNT_OK
            }
            Err(e) => from_client(&e),
        }
    })
}

/// The payloads found, in time order (borrowed: valid until the result is freed; don't free it). Null for null.
///
/// # Safety
///
/// `result` is null or a live result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_query_items(result: *const cnt_query_result) -> *const cnt_items {
    // SAFETY: as the caller promises.
    unsafe { result.as_ref() }.map_or(ptr::null(), |r| &raw const r.items)
}

/// Whether the items cover the whole range. False for null.
///
/// # Safety
///
/// `result` is null or a live result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_query_complete(result: *const cnt_query_result) -> bool {
    // SAFETY: as the caller promises.
    unsafe { result.as_ref() }.is_some_and(|r| r.complete)
}

/// Whether the page was cut short of the range (more may follow the last item). False for null.
///
/// # Safety
///
/// `result` is null or a live result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_query_truncated(result: *const cnt_query_result) -> bool {
    // SAFETY: as the caller promises.
    unsafe { result.as_ref() }.is_some_and(|r| r.truncated)
}

/// How many intervals had nothing. 0 for null.
///
/// # Safety
///
/// `result` is null or a live result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_query_missing_count(result: *const cnt_query_result) -> usize {
    // SAFETY: as the caller promises.
    unsafe { result.as_ref() }.map_or(0, |r| r.missing.len())
}

/// Writes missing interval `index` (UTC ns, half open). `CNT_NOT_FOUND` past the last one.
///
/// # Safety
///
/// `result` is a live result; `start_ns` and `end_ns` are writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_query_missing_at(
    result: *const cnt_query_result,
    index: usize,
    start_ns: *mut u64,
    end_ns: *mut u64,
) -> cnt_status {
    guard(|| {
        // SAFETY: as the caller promises.
        let Some(r) = (unsafe { result.as_ref() }) else { return invalid("result is null") };
        if start_ns.is_null() || end_ns.is_null() {
            return invalid("start_ns or end_ns is null");
        }
        let Some(&(a, b)) = r.missing.get(index) else { return not_found(&format!("no missing interval {index}")) };
        // SAFETY: writable, as the caller promises.
        unsafe {
            *start_ns = a;
            *end_ns = b;
        }
        cnt_status::CNT_OK
    })
}

/// Writes the quote for what's missing. `CNT_NOT_FOUND` when there's none (a hit-only query, nothing missing, or
/// nothing missing can be made).
///
/// # Safety
///
/// `result` is a live result; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_query_quote(result: *const cnt_query_result, out: *mut cnt_promise_quote) -> cnt_status {
    guard(|| {
        // SAFETY: as the caller promises.
        let Some(r) = (unsafe { result.as_ref() }) else { return invalid("result is null") };
        if out.is_null() {
            return invalid("out is null");
        }
        let Some(q) = &r.quote else { return not_found("no quote: nothing missing can be made") };
        let first = q.options.first();
        let got = cnt_promise_quote {
            promise_id: q.promise_id.0,
            status: q.status.0,
            n_options: u32::try_from(q.options.len()).unwrap_or(u32::MAX),
            est_latency_ms: first.map_or(0.0, |o| o.est_latency_ms),
            cost_units: first.map_or(0.0, |o| o.cost_units),
            exceeds_budget: u8::from(first.is_some_and(|o| o.exceeds_budget)),
            shares_in_flight: u8::from(q.shares_in_flight),
            expires_at_ns: q.expires_at_ns,
        };
        // SAFETY: writable, as the caller promises.
        unsafe { *out = got };
        cnt_status::CNT_OK
    })
}

/// Frees a query result (and its items). Null is a no-op.
///
/// # Safety
///
/// `result` is null or a live result, not used again (nor its items).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_query_result_free(result: *mut cnt_query_result) {
    if !result.is_null() {
        // SAFETY: made by cnt_query and not freed before, as the caller promises.
        drop(unsafe { Box::from_raw(result) });
    }
}

/// A committed promise: its outputs as they come. Opaque: made by `cnt_commit_promise`, freed by
/// `cnt_promise_free` (which doesn't cancel it; later outputs then arrive as `CNT_EVENT_PROMISE_FULFILLED`).
pub struct cnt_promise {
    promise: Promise,
    runtime: tokio::runtime::Handle,
}

/// Commits a quoted promise (`option` 0 for the broker's pick, n for its nth alternative, at PriorityBand `band`),
/// writes where it stands to `*life` (PromiseLife: 0 Armed, 1 Dispatching, 2 Cancelled, 3 Failed) if non-null, and
/// a handle its outputs come to, which the caller owns, to `*out`.
///
/// # Safety
///
/// `client` is live; `promise_id` is valid for 16 bytes; `life` is null or writable; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_commit_promise(
    client: *mut cnt_client,
    promise_id: *const u8,
    option: u32,
    band: u8,
    life: *mut u8,
    out: *mut *mut cnt_promise,
) -> cnt_status {
    guard(|| {
        if out.is_null() {
            return invalid("out is null");
        }
        // SAFETY: writable, as the caller promises.
        unsafe { *out = ptr::null_mut() };
        // SAFETY: as the caller promises.
        let Some(c) = (unsafe { client.as_ref() }) else { return invalid("client is null") };
        // SAFETY: as the caller promises.
        let Some(id) = (unsafe { uuid_arg(promise_id) }) else { return invalid("promise_id is null") };
        let band = fb::PriorityBand(band);
        if band.variant_name().is_none() {
            return invalid("band isn't one of the schema's values");
        }
        match c.runtime.block_on(c.client.commit_promise(id, option, band)) {
            Ok((accepted, promise)) => {
                if !life.is_null() {
                    // SAFETY: non-null and writable, as the caller promises.
                    unsafe { *life = accepted.life.0 };
                }
                let handle = cnt_promise { promise, runtime: c.runtime.handle().clone() };
                // SAFETY: writable, checked above.
                unsafe { *out = Box::into_raw(Box::new(handle)) };
                cnt_status::CNT_OK
            }
            Err(e) => from_client(&e),
        }
    })
}

/// Calls a promise off: its jobs are revoked and nothing more is delivered (a held `cnt_promise`'s
/// `cnt_promise_next_part` then answers `CNT_CLOSED`). Writes where it stands to `*life` if non-null. A promise
/// that's gone (fulfilled, failed, or cancelled already) is `CNT_BROKER` with broker code NotFound.
///
/// # Safety
///
/// `client` is live; `promise_id` is valid for 16 bytes; `life` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_cancel_promise(
    client: *mut cnt_client,
    promise_id: *const u8,
    life: *mut u8,
) -> cnt_status {
    guard(|| {
        // SAFETY: as the caller promises.
        let Some(c) = (unsafe { client.as_ref() }) else { return invalid("client is null") };
        // SAFETY: as the caller promises.
        let Some(id) = (unsafe { uuid_arg(promise_id) }) else { return invalid("promise_id is null") };
        match c.runtime.block_on(c.client.cancel_promise(id)) {
            Ok(accepted) => {
                if !life.is_null() {
                    // SAFETY: non-null and writable, as the caller promises.
                    unsafe { *life = accepted.life.0 };
                }
                cnt_status::CNT_OK
            }
            Err(e) => from_client(&e),
        }
    })
}

/// Waits up to `timeout_ms` for the promise's next part of outputs: writes them to `*out` (a list the caller owns)
/// and `*final_part` (nonzero for the last). `CNT_TIMEOUT` if none came; `CNT_CLOSED` once the promise is over (its
/// final part was taken, it was cancelled, or the connection closed); `CNT_BROKER` with the reason if it failed.
///
/// # Safety
///
/// `promise` is a live handle used by no other thread during the call; `out` and `final_part` are writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_promise_next_part(
    promise: *mut cnt_promise,
    timeout_ms: u32,
    out: *mut *mut cnt_items,
    final_part: *mut u8,
) -> cnt_status {
    guard(|| {
        if out.is_null() || final_part.is_null() {
            return invalid("out or final_part is null");
        }
        // SAFETY: writable, as the caller promises.
        unsafe { *out = ptr::null_mut() };
        // SAFETY: as the caller promises.
        let Some(p) = (unsafe { promise.as_mut() }) else { return invalid("promise is null") };
        match within(&p.runtime, timeout_ms, p.promise.next_part()) {
            None => set_error(cnt_status::CNT_TIMEOUT, "no part within the timeout", 0),
            Some(Ok(None)) => set_error(cnt_status::CNT_CLOSED, "the promise is over", 0),
            Some(Ok(Some(part))) => {
                // SAFETY: writable, as the caller promises.
                unsafe {
                    *out = items_handle(part.items);
                    *final_part = u8::from(part.final_part);
                }
                cnt_status::CNT_OK
            }
            Some(Err(e)) => from_client(&e),
        }
    })
}

/// Frees a promise handle (the promise itself goes on). Null is a no-op.
///
/// # Safety
///
/// `promise` is null or a live handle, not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_promise_free(promise: *mut cnt_promise) {
    if !promise.is_null() {
        // SAFETY: made by cnt_commit_promise and not freed before, as the caller promises.
        drop(unsafe { Box::from_raw(promise) });
    }
}

// ---- Subscriptions ----

/// `cnt_subscribe_spec.start`: the next commit on a followed stream, nothing earlier.
pub const CNT_START_LIVE: u8 = 0;
/// `cnt_subscribe_spec.start`: every payload reaching `start_ns` or later, then live.
pub const CNT_START_TIME: u8 = 1;
/// `cnt_subscribe_spec.start`: after the position saved under `durable_name` (everything when none is), then live.
pub const CNT_START_SAVED: u8 = 2;

/// What to follow.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cnt_subscribe_spec {
    /// The streams' names: `n_streams` strings.
    pub streams: *const *const c_char,
    /// How many.
    pub n_streams: usize,
    /// `CNT_START_LIVE`, `CNT_START_TIME`, or `CNT_START_SAVED`.
    pub start: u8,
    /// For `CNT_START_TIME`, UTC ns.
    pub start_ns: u64,
    /// For a durable subscription, the name its position is saved under (null for none).
    pub durable_name: *const c_char,
    /// Nonzero for events with a locator (a pinned copy) as well as the header.
    pub pinned: u8,
}

/// A live subscription. Opaque: made by `cnt_subscribe`; `cnt_subscription_free` ends it.
pub struct cnt_subscription {
    sub: Subscription,
    runtime: tokio::runtime::Handle,
}

/// `cnt_sub_event.type`: a payload (`item`) at `position`.
pub const CNT_SUB_EVENT: u8 = 1;
/// `cnt_sub_event.type`: catch-up is over at `position`; what follows is live.
pub const CNT_SUB_CAUGHT_UP: u8 = 2;
/// `cnt_sub_event.type`: an event at `position` that couldn't be read (`reason` says why).
pub const CNT_SUB_UNDECODABLE: u8 = 3;

/// Something a subscription delivered.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cnt_sub_event {
    /// `CNT_SUB_EVENT`, `CNT_SUB_CAUGHT_UP`, or `CNT_SUB_UNDECODABLE`.
    pub r#type: u8,
    /// Its position (commit sequence).
    pub position: u64,
    /// For `CNT_SUB_EVENT`: a one-item list the caller owns (free it with `cnt_items_free`); null otherwise.
    pub item: *mut cnt_items,
    /// For `CNT_SUB_UNDECODABLE`: why (NUL-terminated, cut to fit).
    pub reason: [c_char; CNT_REASON_MAX],
}

/// Follows `spec`'s streams and writes the subscription to `*out`, which the caller owns.
///
/// # Safety
///
/// `client` is live; `spec`, its arrays, and its strings are valid; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_subscribe(
    client: *mut cnt_client,
    spec: *const cnt_subscribe_spec,
    out: *mut *mut cnt_subscription,
) -> cnt_status {
    guard(|| {
        if out.is_null() {
            return invalid("out is null");
        }
        // SAFETY: writable, as the caller promises.
        unsafe { *out = ptr::null_mut() };
        // SAFETY: as the caller promises.
        let (Some(c), Some(s)) = (unsafe { client.as_ref() }, unsafe { spec.as_ref() }) else {
            return invalid("client or spec is null");
        };
        if s.n_streams == 0 || s.streams.is_null() {
            return invalid("no streams");
        }
        // SAFETY: valid for n_streams pointers, as the caller promises.
        let names = unsafe { std::slice::from_raw_parts(s.streams, s.n_streams) };
        let mut streams = Vec::with_capacity(names.len());
        for &n in names {
            // SAFETY: each a valid string, as the caller promises.
            let Some(name) = (unsafe { str_arg(n) }) else { return invalid("a stream name is null") };
            streams.push(name.to_owned());
        }
        let start = match s.start {
            CNT_START_LIVE => Start::Live,
            CNT_START_TIME => Start::Time(s.start_ns),
            CNT_START_SAVED => Start::Saved,
            _ => return invalid("start must be CNT_START_LIVE, _TIME, or _SAVED"),
        };
        let mut sub = SubscribeSpec::new(streams, start);
        // SAFETY: null or a valid string, as the caller promises.
        sub.durable_name = unsafe { str_arg(s.durable_name) }.map(str::to_owned);
        sub.pinned = s.pinned != 0;
        match c.runtime.block_on(c.client.subscribe(&sub)) {
            Ok(sub) => {
                let handle = cnt_subscription { sub, runtime: c.runtime.handle().clone() };
                // SAFETY: writable, checked above.
                unsafe { *out = Box::into_raw(Box::new(handle)) };
                cnt_status::CNT_OK
            }
            Err(e) => from_client(&e),
        }
    })
}

/// Waits up to `timeout_ms` for the subscription's next delivery and writes it to `*out`. `CNT_TIMEOUT` if none
/// came, `CNT_CLOSED` once it ended (the connection closed).
///
/// # Safety
///
/// `sub` is a live subscription used by no other thread during the call; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_subscription_next(
    sub: *mut cnt_subscription,
    timeout_ms: u32,
    out: *mut cnt_sub_event,
) -> cnt_status {
    guard(|| {
        if out.is_null() {
            return invalid("out is null");
        }
        // SAFETY: as the caller promises.
        let Some(s) = (unsafe { sub.as_mut() }) else { return invalid("sub is null") };
        let mut e = cnt_sub_event { r#type: 0, position: 0, item: ptr::null_mut(), reason: [0; CNT_REASON_MAX] };
        match within(&s.runtime, timeout_ms, s.sub.next()) {
            None => return set_error(cnt_status::CNT_TIMEOUT, "nothing within the timeout", 0),
            Some(None) => return set_error(cnt_status::CNT_CLOSED, "the subscription ended", 0),
            Some(Some(SubEvent::Event { position, item })) => {
                e.r#type = CNT_SUB_EVENT;
                e.position = position;
                e.item = items_handle(vec![item]);
            }
            Some(Some(SubEvent::CaughtUp { position })) => {
                e.r#type = CNT_SUB_CAUGHT_UP;
                e.position = position;
            }
            Some(Some(SubEvent::Undecodable { position, reason })) => {
                e.r#type = CNT_SUB_UNDECODABLE;
                e.position = position;
                copy_reason(&reason, &mut e.reason);
            }
        }
        // SAFETY: writable, as the caller promises.
        unsafe { *out = e };
        cnt_status::CNT_OK
    })
}

/// Says everything up to `position` is handled: a durable subscription resumes after it. One way.
///
/// # Safety
///
/// `sub` is a live subscription.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_subscription_ack(sub: *const cnt_subscription, position: u64) -> cnt_status {
    guard(|| {
        // SAFETY: as the caller promises.
        let Some(s) = (unsafe { sub.as_ref() }) else { return invalid("sub is null") };
        match s.sub.ack(position) {
            Ok(()) => cnt_status::CNT_OK,
            Err(e) => from_client(&e),
        }
    })
}

/// Ends a subscription (without waiting for the broker) and frees it. Null is a no-op.
///
/// # Safety
///
/// `sub` is null or a live subscription, not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_subscription_free(sub: *mut cnt_subscription) {
    let _ = guard(|| {
        if !sub.is_null() {
            // SAFETY: made by cnt_subscribe and not freed before, as the caller promises.
            let s = unsafe { Box::from_raw(sub) };
            let _enter = s.runtime.enter(); // ending it may touch the runtime
            drop(s.sub);
        }
        cnt_status::CNT_OK
    });
}

// ---- Jobs ----

/// A job the broker dispatched (from a `CNT_EVENT_DISPATCH_JOB`). Opaque; freed by `cnt_job_free`. Take its output
/// lease with `cnt_job_take_lease`, read its inputs (`cnt_job_inputs`), and commit the outputs on the lease.
pub struct cnt_job {
    job: DispatchJob,
    inputs: cnt_items,
}

/// What a job asks for.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cnt_job_info {
    /// The job.
    pub job_id: [u8; 16],
    /// The capability's id (from `cnt_register_worker`).
    pub capability_id: u32,
    /// The time to cover, UTC ns, half open.
    pub output_start_ns: u64,
    /// Its end.
    pub output_end_ns: u64,
    /// Its PriorityBand.
    pub band: u8,
    /// When it's due, UTC ns.
    pub deadline_ns: u64,
    /// How many promises it serves.
    pub n_promises: usize,
    /// Nonzero while its output lease is still in the job (not taken, and one came).
    pub has_lease: u8,
}

pub(crate) fn job_handle(mut job: DispatchJob) -> *mut cnt_job {
    let inputs = cnt_items { items: std::mem::take(&mut job.inputs) };
    Box::into_raw(Box::new(cnt_job { job, inputs }))
}

/// Writes what the job asks for to `*out`.
///
/// # Safety
///
/// `job` is a live job; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_job_describe(job: *const cnt_job, out: *mut cnt_job_info) -> cnt_status {
    guard(|| {
        // SAFETY: as the caller promises.
        let Some(j) = (unsafe { job.as_ref() }) else { return invalid("job is null") };
        if out.is_null() {
            return invalid("out is null");
        }
        let d = &j.job;
        let info = cnt_job_info {
            job_id: d.job_id.0,
            capability_id: d.capability_id,
            output_start_ns: d.output_range.0,
            output_end_ns: d.output_range.1,
            band: d.band.0,
            deadline_ns: d.deadline_ns,
            n_promises: d.promise_ids.len(),
            has_lease: u8::from(d.lease.is_some()),
        };
        // SAFETY: writable, as the caller promises.
        unsafe { *out = info };
        cnt_status::CNT_OK
    })
}

/// Writes the id of promise `index` the job serves to `id` (16 bytes). `CNT_NOT_FOUND` past the last one.
///
/// # Safety
///
/// `job` is a live job; `id` is writable for 16 bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_job_promise_at(job: *const cnt_job, index: usize, id: *mut u8) -> cnt_status {
    guard(|| {
        // SAFETY: as the caller promises.
        let Some(j) = (unsafe { job.as_ref() }) else { return invalid("job is null") };
        if id.is_null() {
            return invalid("id is null");
        }
        let Some(p) = j.job.promise_ids.get(index) else { return not_found(&format!("no promise {index}")) };
        // SAFETY: writable for 16 bytes, as the caller promises.
        unsafe { ptr::copy_nonoverlapping(p.0.as_ptr(), id, 16) };
        cnt_status::CNT_OK
    })
}

/// The job's inputs, resolved and pinned by the broker (borrowed: valid until the job is freed; don't free it).
/// Null for null.
///
/// # Safety
///
/// `job` is null or a live job.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_job_inputs(job: *const cnt_job) -> *const cnt_items {
    // SAFETY: as the caller promises.
    unsafe { job.as_ref() }.map_or(ptr::null(), |j| &raw const j.inputs)
}

/// Takes the job's output lease and writes it to `*out` (the caller owns it: commit the outputs, or abort to turn
/// the job down). Once per job: `CNT_PROTOCOL` when it was taken or none came.
///
/// # Safety
///
/// `client` is the live client the job came to; `job` is a live job; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_job_take_lease(
    client: *mut cnt_client,
    job: *mut cnt_job,
    out: *mut *mut cnt_work_lease,
) -> cnt_status {
    guard(|| {
        if out.is_null() {
            return invalid("out is null");
        }
        // SAFETY: writable, as the caller promises.
        unsafe { *out = ptr::null_mut() };
        // SAFETY: as the caller promises.
        let (Some(c), Some(j)) = (unsafe { client.as_ref() }, unsafe { job.as_mut() }) else {
            return invalid("client or job is null");
        };
        match c.client.job_lease(&mut j.job) {
            Ok(lease) => {
                // SAFETY: writable, checked above.
                unsafe { *out = work_lease(lease, c.runtime.handle().clone()) };
                cnt_status::CNT_OK
            }
            Err(e) => from_client(&e),
        }
    })
}

/// Frees a job. An output lease never taken lapses on its own; to turn a job down at once, take its lease and abort
/// it. Null is a no-op.
///
/// # Safety
///
/// `job` is null or a live job, not used again (nor its inputs).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_job_free(job: *mut cnt_job) {
    if !job.is_null() {
        // SAFETY: made by job_handle and not freed before, as the caller promises.
        drop(unsafe { Box::from_raw(job) });
    }
}
