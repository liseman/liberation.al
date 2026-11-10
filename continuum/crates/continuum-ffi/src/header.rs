//! Payload headers for C: making a new payload id, building a PayloadHeader FlatBuffer (what `cnt_ingest` and a
//! commit's outputs take), and reading one back (what a query, subscription, or job hands out), so a C or C++
//! worker needs no FlatBuffers code of its own.

use std::ffi::c_char;

use continuum_core::schema::continuum as fb;
use continuum_core::{HeaderBuf, HeaderSpec, StreamId, Uuid};

use crate::{cnt_status, guard, invalid, set_error, str_arg};

/// Writes a new payload id (a UUIDv7: time ordered) to `id` (16 bytes). Nothing for null.
///
/// # Safety
///
/// `id` is null or writable for 16 bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_new_id(id: *mut u8) {
    if !id.is_null() {
        let new = uuid::Uuid::now_v7().into_bytes();
        // SAFETY: writable for 16 bytes, as the caller promises.
        unsafe { std::ptr::copy_nonoverlapping(new.as_ptr(), id, 16) };
    }
}

/// A payload header to build. Enums are the schema's values.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cnt_header_spec {
    /// The payload's id (`cnt_new_id` makes one).
    pub id: [u8; 16],
    /// Its stream.
    pub stream_id: u32,
    /// The time it covers, UTC ns: exclusive end, or equal to the start for an instant.
    pub start_ns: u64,
    /// Its end.
    pub end_ns: u64,
    /// Order among the stream's payloads with the same start.
    pub seq: u32,
    /// DomainLayer.
    pub layer: u8,
    /// PayloadFormat.
    pub format: u8,
    /// A raw format like "NV12", or a named shape (null for none).
    pub shape_name: *const c_char,
    /// The shape's version.
    pub shape_version: u32,
    /// The payloads it was computed from: `n_parents` ids of 16 bytes, back to back (null if none).
    pub parents: *const u8,
    /// How many.
    pub n_parents: usize,
    /// StorageTier: the coldest tier any input was read from (0 Tier0Hot).
    pub source_tier: u8,
}

/// Builds the header `spec` describes into `out` (`capacity` bytes) and writes its length to `*len`. When it
/// doesn't fit, `CNT_INVALID_ARGUMENT` with `*len` set to the size needed (call again with that much).
///
/// # Safety
///
/// `spec`, its string, and its array are valid; `out` is null (with `capacity` 0) or writable for `capacity` bytes;
/// `len` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_header_build(
    spec: *const cnt_header_spec,
    out: *mut u8,
    capacity: usize,
    len: *mut usize,
) -> cnt_status {
    guard(|| {
        // SAFETY: as the caller promises.
        let Some(s) = (unsafe { spec.as_ref() }) else { return invalid("spec is null") };
        if len.is_null() {
            return invalid("len is null");
        }
        let (layer, format, tier) =
            (fb::DomainLayer(s.layer), fb::PayloadFormat(s.format), fb::StorageTier(s.source_tier));
        if layer.variant_name().is_none() || format.variant_name().is_none() || tier.variant_name().is_none() {
            return invalid("layer, format, or source_tier isn't one of the schema's values");
        }
        let parents = if s.n_parents == 0 {
            Vec::new()
        } else if s.parents.is_null() {
            return invalid("parents is null");
        } else {
            let Some(n) = s.n_parents.checked_mul(16) else { return invalid("n_parents is too large") };
            // SAFETY: valid for 16 * n_parents bytes, as the caller promises.
            let raw = unsafe { std::slice::from_raw_parts(s.parents, n) };
            raw.chunks_exact(16).map(|c| Uuid(c.try_into().expect("16-byte chunks"))).collect()
        };
        let mut h = HeaderSpec::raw(Uuid(s.id), StreamId(s.stream_id), s.start_ns, s.end_ns);
        h.seq = s.seq;
        h.layer = layer;
        h.format = format;
        // SAFETY: null or a valid string, as the caller promises.
        h.shape_name = unsafe { str_arg(s.shape_name) }.unwrap_or("").to_owned();
        h.shape_version = s.shape_version;
        h.parents = parents;
        h.source_tier = tier;
        let built = match HeaderBuf::build(&h) {
            Ok(b) => b,
            Err(e) => return invalid(&format!("header: {e}")),
        };
        let bytes = built.as_bytes();
        // SAFETY: writable, as the caller promises.
        unsafe { *len = bytes.len() };
        if bytes.len() > capacity || out.is_null() {
            return set_error(
                cnt_status::CNT_INVALID_ARGUMENT,
                &format!("the header needs {} bytes; out has {capacity}", bytes.len()),
                0,
            );
        }
        // SAFETY: writable for capacity >= bytes.len() bytes, as the caller promises.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len()) };
        cnt_status::CNT_OK
    })
}

/// What a header says, as C reads it.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cnt_header_info {
    /// The payload's id.
    pub id: [u8; 16],
    /// Its stream.
    pub stream_id: u32,
    /// The time it covers, UTC ns.
    pub start_ns: u64,
    /// Its end.
    pub end_ns: u64,
    /// Its seq.
    pub seq: u32,
    /// DomainLayer.
    pub layer: u8,
    /// PayloadFormat.
    pub format: u8,
    /// How many parents it names.
    pub n_parents: usize,
}

/// Reads a PayloadHeader FlatBuffer (`len` bytes, like a `cnt_item`'s header) into `*out`.
///
/// # Safety
///
/// `header` is valid for `len` bytes; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnt_header_read(header: *const u8, len: usize, out: *mut cnt_header_info) -> cnt_status {
    guard(|| {
        if header.is_null() || out.is_null() {
            return invalid("header or out is null");
        }
        // SAFETY: valid for len bytes, as the caller promises.
        let bytes = unsafe { std::slice::from_raw_parts(header, len) }.to_vec();
        let h = match HeaderBuf::new(bytes) {
            Ok(h) => h,
            Err(e) => return invalid(&format!("header: {e}")),
        };
        let v = h.header();
        let info = cnt_header_info {
            id: h.id().0,
            stream_id: v.stream_id(),
            start_ns: v.start_ns(),
            end_ns: v.end_ns(),
            seq: v.seq(),
            layer: v.layer().0,
            format: v.format().0,
            n_parents: v.parents().map_or(0, |p| p.len()),
        };
        // SAFETY: writable, as the caller promises.
        unsafe { *out = info };
        cnt_status::CNT_OK
    })
}
