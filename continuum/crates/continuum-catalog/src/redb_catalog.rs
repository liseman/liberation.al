//! [`RedbCatalog`]: the [`Catalog`] contract in a redb file.
//!
//! Reads open their own snapshot without holding the writer lock, so the broker can run a pool of reader threads.
//! Time-ordered pages seek to their full time/sequence cursor, so a complete startup scan visits each row once
//! apart from each page's cursor and lookahead, rather than revisiting the growing prefix on every page.
//! Window pages share a bounded per-window cache; different windows scan independently. Writes are short transactions; a purge and a scrub are jobs of bounded steps, so the broker's
//! writer thread can put ingest commits between them. The database sits behind a lock that a scrub takes for one
//! moment only, to put its fresh file in place; a reader that began before keeps reading its old snapshot.

use std::collections::{BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock, Weak};

use continuum_core::{BootId, HeaderBuf, Interval, StageId, StorageLocator, StorageTier, StreamId, Uuid};
use redb::{Database, Durability, ReadableDatabase};

use crate::codec::{self, In, Out};
use crate::store::{R, Reader, W, prefix_end};
use crate::tables::*;
use crate::types::*;
use crate::{Catalog, CatalogError, MAX_CHUNK_FRAMES, PURGE_BATCH_ROWS, Result, WorkerState, worker_state};

/// A scrub's catch-up steps run outside every lock until at most this many written keys are left to carry; only
/// those go into the swap, which holds the catalog alone. So the swap's length doesn't grow with the catalog's size
/// or the step budget: it is these few keys, one commit, and a rename.
pub const SWAP_DELTA: usize = 64;

/// The layout of the catalog file's tables and records, stamped in META at creation. Bump it with any change to
/// how a record is encoded or which tables a file holds; [`RedbCatalog::open`] refuses a file in another format, and
/// the fix is a fresh file (`pack-reset-catalog`). There are no upgrades.
pub const FORMAT: u64 = 8;

/// The first commit sequence of the recording era, set at creation: every payload a format 8 file holds is in it.
const RECORDING_ERA_START: u64 = 1;

// ---- keys ----

fn recording_era(r: &impl Reader) -> Result<u64> {
    r.get(META, b"recording_era_start_seq")?
        .ok_or_else(|| CatalogError::Storage("the catalog is missing its recording era".into()))
        .and_then(|b| codec::u64_at(&b, 0))
}
fn recording_init_key(stream: StreamId) -> Vec<u8> {
    Out::new().raw(b"recording_initialized:").u32(stream.0).done()
}
fn recording_meta(r: &impl Reader, stream: StreamId) -> Result<crate::recording::Meta> {
    if let Some(bytes) = r.get(RECORDING_META, &k32(stream.0))? {
        return crate::recording::Meta::decode(&bytes);
    }
    if r.has(META, &recording_init_key(stream))? {
        return Err(crate::recording::invalid("missing initialized source metadata"));
    }
    Ok(crate::recording::Meta::default())
}
/// An instant audio original still holding its stream's history unknown.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstantOriginal {
    /// Its stream.
    pub stream: StreamId,
    /// The original's id.
    pub original: Uuid,
    /// Its header's start.
    pub start_ns: u64,
    /// The chunk and frame index holding its samples, once flushed.
    pub frame: Option<(Uuid, u32)>,
}

/// What the open's recording repair did: the streams it restored, and each stream it held back with why.
type RecordingRepair = (Vec<u32>, Vec<(u32, String)>);

/// Whether a ledger row is still a reason its stream's current-era history can't be known: a raw original whose
/// subjects overflowed, or an instant audio original (no source extent; it indexes as at most a nanosecond).
fn recording_reason(row: &crate::recording::Row, audio: bool) -> bool {
    !row.subjects_complete || (audio && !row.scope_only && !row.bound && row.input.spans.is_empty() && row.reach() <= 1)
}

/// Once, at open: a stream whose current-era history is marked unknown gets it back when none of its rows is
/// still a reason (`recording_reason`). Nothing else ever resets the mark, so a stream stayed unknown after the
/// original that caused it had gone, and with it every camera whose aging reads that microphone.
///
/// An instant audio original absorbed into a chunk keeps its row: absorption removes only rows that are neither
/// bound nor unknown, and an instant's row is unknown (`Row::raw`), which the chunk's row inherits. So its reason
/// stays until the original itself is gone. A scope-only row landing on a real row's key, the third cause, leaves no
/// row; it needs a purge id equal to a stored payload's at the same start, which generated ids rule out, and a stream
/// a live range purge still has to sweep waits for a later open. Streams never initialized are left alone.
///
/// Restoring only needs readers to re-read, so the stream's invalidation revision is bumped; nothing was forgotten,
/// so its forget mark and the worker states built from it stay (`invalidate` is for the other direction). Returns the
/// streams restored.
fn recording_repair_known(w: &mut W) -> Result<RecordingRepair> {
    let swept = recording_swept(w)?;
    let mut unknown = Vec::new();
    w.scan(RECORDING_META, &[], None, &mut |k, v| {
        if !crate::recording::Meta::decode(v)?.current_known {
            unknown.push(codec::u32_at(k, 0)?);
        }
        Ok(true)
    })?;
    let (mut restored, mut held) = (Vec::new(), Vec::new());
    for stream in unknown {
        match recording_restore_one(w, stream, &swept)? {
            Restore::Restored => restored.push(stream),
            Restore::Held(why) => held.push((stream, why)),
            Restore::NotNative => {}
        }
    }
    Ok((restored, held))
}

/// Streams a live range purge still has to sweep, with that purge. Only those wait: worker checkpoints and face.live
/// are range-purged all the time, and a stale entry whose job is gone sweeps nothing.
fn recording_swept(w: &W) -> Result<std::collections::BTreeMap<u32, Uuid>> {
    let mut sweeping = Vec::new();
    w.scan(RECORDING_PURGE, &[], None, &mut |k, v| {
        if v.first() != Some(&1) {
            sweeping.push(Uuid(k.try_into().map_err(|_| crate::recording::invalid("invalid purge key"))?));
        }
        Ok(true)
    })?;
    let mut swept = std::collections::BTreeMap::new();
    for purge in sweeping {
        if let Some(job) = load_job(w, purge)?
            && matches!(job.selector, PurgeSelector::Range { .. })
        {
            for stream in job.streams {
                swept.entry(stream).or_insert(purge);
            }
        }
    }
    Ok(swept)
}

enum Restore {
    Restored,
    Held(String),
    NotNative,
}

/// Restores one unknown stream's current-era history when nothing still holds it (`recording_repair_known`).
fn recording_restore_one(w: &mut W, stream: u32, swept: &std::collections::BTreeMap<u32, Uuid>) -> Result<Restore> {
    let Some(info) = w.get(STREAMS, &k32(stream))?.map(|b| codec::read_stream_info(&b)).transpose()? else {
        return Ok(Restore::Held("not a registered stream".into()));
    };
    if !crate::recording::native(&info) {
        return Ok(Restore::NotNative);
    }
    if let Some(purge) = swept.get(&stream) {
        return Ok(Restore::Held(format!("range purge {purge} still has to sweep it")));
    }
    if !w.has(META, &recording_init_key(StreamId(stream)))? {
        return Ok(Restore::Held("never initialized".into()));
    }
    let audio = info.name.ends_with(".audio");
    let mut reason = None;
    w.prefix(RECORDING_ROWS, &Out::new().u32(stream).done(), |_, v| {
        let row = crate::recording::Row::decode(v)?;
        if recording_reason(&row, audio) {
            reason = Some(format!(
                "original {} at {} ns is still a reason ({})",
                row.input.chunk_id,
                row.input.start_ns,
                if row.subjects_complete { "an instant audio header" } else { "its subjects overflowed" }
            ));
        }
        Ok(reason.is_none())
    })?;
    if let Some(reason) = reason {
        return Ok(Restore::Held(reason));
    }
    let mut meta = recording_meta(w, StreamId(stream))?;
    if !meta.current_known {
        meta.current_known = true;
        recording_save_meta(w, StreamId(stream), &meta)?;
        bump(w, StreamId(stream))?;
    }
    Ok(Restore::Restored)
}

fn recording_save_meta(w: &mut W, stream: StreamId, meta: &crate::recording::Meta) -> Result<()> {
    if !meta.current_known && recording_meta(w, stream)?.current_known {
        invalidate(w, stream)?;
    }
    w.put(RECORDING_META, &k32(stream.0), &meta.encode())?;
    w.put(META, &recording_init_key(stream), &[1])
}
fn recording_key(stream: StreamId, row: &crate::recording::Row) -> Vec<u8> {
    Out::new().u32(stream.0).u64(row.cursor().start_ns).id(row.input.chunk_id).done()
}
fn recording_get(r: &impl Reader, id: Uuid) -> Result<Option<crate::recording::Row>> {
    let Some(key) = r.get(RECORDING_IDS, &id.0)? else { return Ok(None) };
    let bytes = r.get(RECORDING_ROWS, &key)?.ok_or_else(|| crate::recording::invalid("missing recording row"))?;
    let row = crate::recording::Row::decode(&bytes)?;
    if key.len() != 28
        || row.input.chunk_id != id
        || row.scope_only
        || recording_key(StreamId(codec::u32_at(&key, 0)?), &row) != key
    {
        return Err(crate::recording::invalid("recording identity index disagrees"));
    }
    for subject in &row.subjects {
        if !r.has(RECORDING_SUBJECTS, &k_pair(*subject, id))? {
            return Err(crate::recording::invalid("missing original subject index"));
        }
    }
    Ok(Some(row))
}
fn recording_put(w: &mut W, stream: StreamId, row: crate::recording::Row) -> Result<()> {
    let key = recording_key(stream, &row);
    if row.scope_only {
        if w.get(RECORDING_ROWS, &key)?
            .map(|b| crate::recording::Row::decode(&b))
            .transpose()?
            .is_some_and(|r| !r.scope_only)
        {
            let mut meta = recording_meta(w, stream)?;
            meta.known = false;
            meta.current_known = false;
            recording_save_meta(w, stream, &meta)?;
            return Ok(());
        }
    } else {
        recording_remove(w, row.input.chunk_id)?;
    }
    let mut meta = recording_meta(w, stream)?;
    meta.longest = meta.longest.max(row.reach());
    if !row.subjects_complete {
        meta.known = false;
        meta.current_known = false;
    }
    recording_save_meta(w, stream, &meta)?;
    if !row.scope_only {
        w.put(RECORDING_IDS, &row.input.chunk_id.0, &key)?;
        for subject in &row.subjects {
            w.put(RECORDING_SUBJECTS, &k_pair(*subject, row.input.chunk_id), &[])?;
        }
    }
    w.put(RECORDING_ROWS, &key, &row.encode())
}
fn recording_remove(w: &mut W, id: Uuid) -> Result<()> {
    if let Some(old) = w.del(RECORDING_IDS, &id.0)?
        && let Some(bytes) = w.del(RECORDING_ROWS, &old)?
    {
        for subject in crate::recording::Row::decode(&bytes)?.subjects {
            w.del(RECORDING_SUBJECTS, &k_pair(subject, id))?;
        }
    }
    Ok(())
}
fn recording_scope(w: &mut W, stream: StreamId, span: Interval, purge: Uuid, first_seq: u64) -> Result<()> {
    let Some(info) = w
        .get(STREAMS, &k32(stream.0))?
        .map(|b| codec::read_stream_info(&b))
        .transpose()?
        .filter(crate::recording::native)
    else {
        return Ok(());
    };
    let header = HeaderBuf::build(&continuum_core::HeaderSpec::raw(purge, stream, span.start_ns(), span.end_ns()))
        .map_err(bad_header)?;
    let mut row = crate::recording::Row::raw(&header, &info, recording_meta(w, stream)?.generation, first_seq)?;
    row.unknown = true;
    row.scope_only = true;
    recording_put(w, stream, row)?;
    invalidate(w, stream)
}
// Complete each bounded ledger sweep before the ordinary purge can remove source rows. Pending scope rows
// block publication immediately; after sweeping, only affected current-era witnesses veto current-era targets.
fn recording_range_step(w: &mut W, purge: Uuid, job: &mut Job, budget: usize) -> Result<bool> {
    let PurgeSelector::Range { span, .. } = job.selector else { return Ok(true) };
    let Some(bytes) = w.get(RECORDING_PURGE, &purge.0)? else { return Ok(true) };
    let mut input = In::new(&bytes);
    if input.u8()? == 1 {
        return Ok(true);
    }
    let cutoff = input.u64()?;
    let mut position = input.u32()? as usize;
    let mut after = input.bytes()?.to_vec();
    let mut left = budget.min(crate::MAX_RECORDING_SCAN);
    let mut visited = 0;
    while left > 0 && position < job.streams.len() {
        let stream = StreamId(job.streams[position]);
        let meta = recording_meta(w, stream)?;
        let lo = if after.is_empty() {
            Out::new().u32(stream.0).u64(span.start_ns().saturating_sub(meta.longest)).done()
        } else {
            after.clone()
        };
        let hi = Out::new().u32(stream.0).u64(span.end_ns()).done();
        let mut rows = Vec::new();
        w.scan(RECORDING_ROWS, &lo, Some(&hi), &mut |key, value| {
            if key != after {
                rows.push((key.to_vec(), crate::recording::Row::decode(value)?));
            }
            Ok(rows.len() < left)
        })?;
        let full = rows.len() == left;
        // Empty sources are metadata-only cursor advances, not purge work. In
        // particular, non-media purges must retain their original first-step
        // payload progress and cooperative deadline accounting.
        visited += rows.len();
        left = left.saturating_sub(rows.len());
        job.done = job.done.saturating_add(rows.len() as u64);
        for (key, row) in rows {
            let overlaps = if row.input.spans.is_empty() {
                row.input.start_ns < span.end_ns() && row.end_ns > span.start_ns()
            } else {
                row.input.spans.iter().any(|s| s.overlaps(span))
            };
            if !row.scope_only && row.input.first_seq >= recording_era(w)? && row.input.first_seq <= cutoff && overlaps
            {
                recording_unknown(w, row.input.chunk_id)?;
            }
            after = key;
        }
        if !full {
            let key = Out::new().u32(stream.0).u64(span.start_ns()).id(purge).done();
            if let Some(bytes) = w.get(RECORDING_ROWS, &key)?
                && crate::recording::Row::decode(&bytes)?.scope_only
            {
                w.del(RECORDING_ROWS, &key)?;
                invalidate(w, stream)?;
            }
            position += 1;
            after.clear();
        }
    }
    let done = position >= job.streams.len();
    w.put(
        RECORDING_PURGE,
        &purge.0,
        &Out::new().u8(u8::from(done)).u64(cutoff).u32(position as u32).bytes(&after).done(),
    )?;
    Ok(done && visited == 0)
}

fn recording_unknown(w: &mut W, id: Uuid) -> Result<()> {
    let owner = frame_row(w, id)?.map_or(id, |(chunk, _, _)| chunk);
    if let Some(mut row) = recording_get(w, owner)? {
        let changed = !row.unknown;
        row.unknown = true;
        let key =
            w.get(RECORDING_IDS, &owner.0)?.ok_or_else(|| crate::recording::invalid("missing original identity"))?;
        w.put(RECORDING_ROWS, &key, &row.encode())?;
        if changed {
            invalidate(w, StreamId(codec::u32_at(&key, 0)?))?;
        }
    } else if let Some(h) = header_of(w, owner)?
        && let Some(info) = w.get(STREAMS, &k32(h.stream().0))?.map(|b| codec::read_stream_info(&b)).transpose()?
        && crate::recording::native(&info)
    {
        let mut row = crate::recording::Row::raw(
            &h,
            &info,
            recording_meta(w, h.stream())?.generation,
            w.get(PAYLOAD_SEQ, &h.id().0)?.map_or(Ok(0), |b| codec::u64_at(&b, 0))?,
        )?;
        row.unknown = true;
        recording_put(w, h.stream(), row)?;
        invalidate(w, h.stream())?;
    }
    Ok(())
}

fn k32(v: u32) -> Vec<u8> {
    v.to_be_bytes().to_vec()
}
fn k_time(stream: u32, start: u64, seq: u32) -> Vec<u8> {
    Out::new().u32(stream).u64(start).u32(seq).done()
}
fn k_pair(a: Uuid, b: Uuid) -> Vec<u8> {
    Out::new().id(a).id(b).done()
}
fn k_loc(id: Uuid, tier: StorageTier) -> Vec<u8> {
    Out::new().id(id).u8(tier.0).done()
}

fn k_cov(source: u32, stage: u32, start: u64) -> Vec<u8> {
    Out::new().u32(source).u32(stage).u64(start).done()
}
fn k_seq(purge: Uuid, seq: u64) -> Vec<u8> {
    Out::new().id(purge).u64(seq).done()
}
fn k_cut(chunk: Uuid, purge: Uuid, index: u32) -> Vec<u8> {
    Out::new().id(chunk).id(purge).u32(index).done()
}
fn k_cut_order(purge: Uuid, chunk: Uuid, index: u32) -> Vec<u8> {
    Out::new().id(purge).id(chunk).u32(index).done()
}

fn invalidation_key(stream: StreamId) -> Vec<u8> {
    Out::new().raw(b"invalidation:").u32(stream.0).done()
}

/// Where a stream's last forget left its revision. Absent until the first age-out after this was built: then
/// every earlier change counts as a forget, so the current revision is the mark.
fn forgotten_key(stream: StreamId) -> Vec<u8> {
    Out::new().raw(b"forgotten_at:").u32(stream.0).done()
}

/// A change to what a stream serves that forgets something: a purge, a cut, a ledger or retention change. Readers'
/// caches, dependency binding, and worker state all see it.
fn invalidate(w: &mut W, stream: StreamId) -> Result<()> {
    let revision = bump(w, stream)?;
    w.put(META, &forgotten_key(stream), &revision.to_be_bytes())?;
    // A worker's state built from this stream may hold what changed: it goes in the same transaction.
    let mut workers = Vec::new();
    w.prefix(WORKER_STATE_INPUTS, &worker_state::input_prefix(stream), |k, _| {
        workers.push(k[4..].to_vec());
        Ok(true)
    })?;
    for worker in workers {
        drop_worker_state(w, &worker)?;
    }
    Ok(())
}

/// An age-out's change: readers' caches see it, but aging isn't forgetting, so what was built from the aged
/// rows stays right: a dependency or worker state read before it is still good.
fn invalidate_aged(w: &mut W, stream: StreamId) -> Result<()> {
    let key = forgotten_key(stream);
    if !w.has(META, &key)? {
        let before = revision_of(w, stream)?;
        w.put(META, &key, &before.to_be_bytes())?;
    }
    bump(w, stream).map(|_| ())
}

/// Advances the stream's revision, returning the new one. A saturated revision is advertised as unsupported,
/// preserving conservative reader invalidation.
fn bump(w: &mut W, stream: StreamId) -> Result<u64> {
    let revision = revision_of(w, stream)?.saturating_add(1);
    w.put(META, &invalidation_key(stream), &revision.to_be_bytes())?;
    Ok(revision)
}

/// Whether `revision`, read from `stream` earlier, is still good for binding a dependency or a worker state: reached,
/// not exhausted, and with no forget since (age-outs may have moved it on).
fn unforgotten_since(r: &impl Reader, stream: StreamId, revision: u64) -> Result<bool> {
    let now = revision_of(r, stream)?;
    let mark = r.get(META, &forgotten_key(stream))?.map_or(Ok(now), |v| codec::u64_at(&v, 0))?;
    Ok(now != u64::MAX && mark <= revision && revision <= now)
}

/// The stored revision of a stream: what `invalidate` has advanced it to, 0 for one never invalidated.
fn revision_of(r: &impl Reader, stream: StreamId) -> Result<u64> {
    r.get(META, &invalidation_key(stream))?.map_or(Ok(0), |v| codec::u64_at(&v, 0))
}

/// Deletes one worker's state row and its input index entries; whether it had one.
fn drop_worker_state(w: &mut W, worker: &[u8]) -> Result<bool> {
    let Some(value) = w.del(WORKER_STATE, worker)? else {
        return Ok(false);
    };
    let (owner, contract) = worker_state::split_key(worker)?;
    for id in worker_state::decode(&owner, &contract, &value)?.ids {
        w.del(WORKER_STATE_INPUTS, &worker_state::input_key(id, worker))?;
    }
    Ok(true)
}

fn bad_header(e: continuum_core::DecodeError) -> CatalogError {
    CatalogError::Storage(Box::new(e))
}

fn put_location(w: &mut W, id: Uuid, loc: &StorageLocator) -> Result<()> {
    w.put(LOCATIONS, &k_loc(id, loc.tier()), &loc.encode())
}
fn del_location(w: &mut W, id: Uuid, tier: StorageTier) -> Result<()> {
    w.del(LOCATIONS, &k_loc(id, tier))?;
    Ok(())
}

fn header_of(rd: &impl Reader, id: Uuid) -> Result<Option<HeaderBuf>> {
    rd.get(PAYLOADS, &id.0)?.map(|b| HeaderBuf::new(b).map_err(bad_header)).transpose()
}

fn span_of(h: &HeaderBuf) -> Result<Interval> {
    h.span().ok_or_else(|| CatalogError::Invalid(format!("payload {} is an instant at u64::MAX", h.id())))
}

fn known(rd: &impl Reader, id: Uuid) -> Result<bool> {
    Ok(rd.has(PAYLOADS, &id.0)? || rd.has(FRAMES, &id.0)?)
}

fn doomed(rd: &impl Reader, id: Uuid) -> Result<bool> {
    rd.has(DOOMED, &id.0)
}

fn locations(rd: &impl Reader, id: Uuid) -> Result<Vec<StorageLocator>> {
    let mut out = Vec::new();
    rd.prefix(LOCATIONS, &id.0, |_, v| {
        out.push(codec::locator(v)?);
        Ok(true)
    })?;
    Ok(out)
}

fn chunk_frames(rd: &impl Reader, chunk: Uuid) -> Result<Option<Vec<ChunkFrame>>> {
    rd.get(CHUNKS, &chunk.0)?.map(|b| codec::read_frames(&b)).transpose()
}

fn indexed(rd: &impl Reader, id: Uuid) -> Result<Option<Indexed>> {
    if doomed(rd, id)? {
        return Ok(None);
    }
    let Some(header) = header_of(rd, id)? else { return Ok(None) };
    indexed_header(rd, id, header).map(Some)
}

/// Resolve the remaining fields using a header already validated in this same read snapshot.
fn indexed_header(rd: &impl Reader, id: Uuid, header: HeaderBuf) -> Result<Indexed> {
    let commit_seq = rd.get(PAYLOAD_SEQ, &id.0)?.map_or(Ok(0), |b| codec::u64_at(&b, 0))?;
    Ok(Indexed { header, locations: locations(rd, id)?, frames: chunk_frames(rd, id)?, commit_seq })
}

/// One time-ordered catalog page, separated from snapshot creation for counted-reader regressions.
fn range_page(
    r: &impl Reader,
    stream: StreamId,
    window: Interval,
    after: Option<RangeCursor>,
    limit: usize,
) -> Result<Page> {
    if limit == 0 {
        return Err(CatalogError::Invalid("a range's limit must be at least 1".into()));
    }
    if !r.has(STREAMS, &k32(stream.0))? {
        return Err(CatalogError::NotFound(format!("stream {}", stream.0)));
    }
    let reach = r.get(STREAM_LONGEST, &k32(stream.0))?.map_or(Ok(0), |b| codec::u64_at(&b, 0))?;
    let mut lo = k_time(stream.0, window.start_ns().saturating_sub(reach), 0);
    if let Some(cursor) = after {
        // Seek to the cursor's complete time/sequence key. The scan is inclusive; the callback excludes
        // that one key without incrementing a maximum sequence or skipping ties at the same time.
        lo = lo.max(k_time(stream.0, cursor.start_ns, cursor.seq));
    }
    let hi = k_time(stream.0, window.end_ns(), 0);
    if lo >= hi {
        return Ok(Page { items: Vec::new(), next: None });
    }
    let mut items = Vec::new();
    let mut last = None;
    let mut more = false;
    r.scan(TIME_INDEX, &lo, Some(&hi), &mut |k, v| {
        let (start, seq) = (codec::u64_at(k, 4)?, codec::u32_at(k, 12)?);
        if after.is_some_and(|a| (start, seq) <= (a.start_ns, a.seq)) {
            return Ok(true);
        }
        let id = codec::id_at(v, 0)?;
        // The run-up before the window holds mostly payloads that end before it: check the span first.
        if !header_of(r, id)?.and_then(|h| h.span()).is_some_and(|s| s.overlaps(window)) {
            return Ok(true);
        }
        let Some(p) = indexed(r, id)? else { return Ok(true) };
        if items.len() == limit {
            more = true;
            return Ok(false);
        }
        items.push(p);
        last = Some(RangeCursor { start_ns: start, seq });
        Ok(true)
    })?;
    Ok(Page { items, next: if more { last } else { None } })
}

/// An absorbed frame's (chunk, index, subjects).
fn frame_row(rd: &impl Reader, id: Uuid) -> Result<Option<(Uuid, u32, Vec<Uuid>)>> {
    let Some(b) = rd.get(FRAMES, &id.0)? else { return Ok(None) };
    let mut i = In::new(&b);
    let chunk = i.id()?;
    let index = i.u32()?;
    Ok(Some((chunk, index, codec::ids(i.rest())?)))
}

/// The time frame `index` of `chunk` holds, on the chunk's stream (see the trait's purge docs).
fn frame_slot(rd: &impl Reader, chunk: Uuid, index: u32) -> Result<Option<(StreamId, Interval)>> {
    let (Some(h), Some(frames)) = (header_of(rd, chunk)?, chunk_frames(rd, chunk)?) else { return Ok(None) };
    let Some(span) = h.span() else { return Ok(None) };
    Ok(frame_slots(span, &frames).get(index as usize).map(|&slot| (h.stream(), slot)))
}

fn doomed_entry(rd: &impl Reader, id: Uuid) -> Result<Option<Doomed>> {
    if let Some(h) = header_of(rd, id)? {
        let tiers = locations(rd, id)?.iter().map(StorageLocator::tier).collect();
        return Ok(h.span().map(|span| Doomed { id, stream: h.stream(), span, tiers, late: false }));
    }
    let Some((chunk, index, _)) = frame_row(rd, id)? else { return Ok(None) };
    Ok(frame_slot(rd, chunk, index)?.map(|(stream, span)| Doomed { id, stream, span, tiers: Vec::new(), late: false }))
}

fn children(rd: &impl Reader, id: Uuid) -> Result<Vec<Uuid>> {
    let mut out = Vec::new();
    rd.prefix(PROV_CHILDREN, &id.0, |k, _| {
        out.push(codec::id_at(k, 16)?);
        Ok(true)
    })?;
    Ok(out)
}

fn parents(rd: &impl Reader, id: Uuid) -> Result<Vec<Uuid>> {
    rd.get(PROV_PARENTS, &id.0)?.map_or(Ok(Vec::new()), |b| codec::ids(&b))
}

/// The chunk's absorbed frames that kept a row.
fn kept_frames(rd: &impl Reader, chunk: Uuid) -> Result<Vec<Uuid>> {
    let mut out = Vec::new();
    rd.prefix(CHUNK_FRAMES, &chunk.0, |k, _| {
        out.push(codec::id_at(k, 16)?);
        Ok(true)
    })?;
    Ok(out)
}

// ---- coverage on the COVERAGE table ----

fn cov_within(rd: &impl Reader, source: u32, stage: u32, window: Interval) -> Result<Vec<Interval>> {
    let prefix = Out::new().u32(source).u32(stage).done();
    let mut out = Vec::new();
    let mut push = |s: u64, e: u64| {
        if let Some(i) = Interval::new(s.max(window.start_ns()), e.min(window.end_ns())) {
            out.push(i);
        }
    };
    if let Some((k, v)) = rd.last_below(COVERAGE, &prefix, &k_cov(source, stage, window.start_ns()))? {
        push(codec::u64_at(&k, 8)?, codec::u64_at(&v, 0)?);
    }
    let mut rows = Vec::new();
    rd.scan(
        COVERAGE,
        &k_cov(source, stage, window.start_ns()),
        Some(&k_cov(source, stage, window.end_ns())),
        &mut |k, v| {
            rows.push((codec::u64_at(k, 8)?, codec::u64_at(v, 0)?));
            Ok(true)
        },
    )?;
    for (s, e) in rows {
        push(s, e);
    }
    Ok(out)
}

fn cov_gaps(rd: &impl Reader, source: u32, stage: u32, window: Interval, max: usize) -> Result<Vec<Interval>> {
    let mut set = crate::CoverageSet::new();
    for i in cov_within(rd, source, stage, window)? {
        set.insert(i);
    }
    Ok(set.gaps(window, max))
}

fn k_dep(stream: u32, start: u64, id: Uuid) -> Vec<u8> {
    Out::new().u32(stream).u64(start).id(id).done()
}

fn k_dep_of(id: Uuid, stream: u32, start: u64) -> Vec<u8> {
    Out::new().id(id).u32(stream).u64(start).done()
}

/// Stores `id`'s dependencies, each one row by (stream, start) for a purge's overlap scan and one by `id` to
/// drop them when it goes.
fn put_dependencies(w: &mut W, id: Uuid, deps: &[Dependency]) -> Result<()> {
    for d in deps {
        let stream = d.stream.0;
        if !w.has(STREAMS, &k32(stream))? {
            return Err(CatalogError::NotFound(format!("dependency on stream {stream}, which isn't registered")));
        }
        if !unforgotten_since(w, d.stream, d.revision)? {
            let now = revision_of(w, d.stream)?;
            return Err(CatalogError::Conflict(format!(
                "stream {stream} changed since its revision {} was read (now {now}): check what was used again",
                d.revision
            )));
        }
        let (start, end) = (d.span.start_ns(), d.span.end_ns());
        w.put(DEPS, &k_dep(stream, start, id), &end.to_be_bytes())?;
        w.put(DEPS_OF, &k_dep_of(id, stream, start), &[])?;
        let longest = w.get(DEPS_LONGEST, &k32(stream))?.map_or(Ok(0), |b| codec::u64_at(&b, 0))?;
        if end - start > longest {
            w.put(DEPS_LONGEST, &k32(stream), &(end - start).to_be_bytes())?;
        }
    }
    Ok(())
}

/// The payloads with a dependency on `stream` overlapping `span`.
fn dependents(rd: &impl Reader, stream: u32, span: Interval) -> Result<Vec<Uuid>> {
    let longest = rd.get(DEPS_LONGEST, &k32(stream))?.map_or(Ok(0), |b| codec::u64_at(&b, 0))?;
    if longest == 0 {
        return Ok(Vec::new());
    }
    let lo = k_dep(stream, span.start_ns().saturating_sub(longest), Uuid([0; 16]));
    let hi = k_dep(stream, span.end_ns(), Uuid([0; 16]));
    let mut out = Vec::new();
    rd.scan(DEPS, &lo, Some(&hi), &mut |k, v| {
        if codec::u64_at(v, 0)? > span.start_ns() {
            out.push(codec::id_at(k, 12)?);
        }
        Ok(true)
    })?;
    Ok(out)
}

/// `id`'s stored dependencies, as (stream, start, end), sorted.
fn dependencies_of(rd: &impl Reader, id: Uuid) -> Result<Vec<(u32, u64, u64)>> {
    let mut keys = Vec::new();
    rd.prefix(DEPS_OF, &id.0, |k, _| {
        keys.push((codec::u32_at(k, 16)?, codec::u64_at(k, 20)?));
        Ok(true)
    })?;
    let mut out = Vec::new();
    for (stream, start) in keys {
        let end = rd.get(DEPS, &k_dep(stream, start, id))?.map_or(Ok(0), |b| codec::u64_at(&b, 0))?;
        out.push((stream, start, end));
    }
    out.sort_unstable();
    Ok(out)
}

/// Drops `id`'s dependencies: it went.
fn drop_dependencies(w: &mut W, id: Uuid) -> Result<()> {
    let mut rows = Vec::new();
    w.prefix(DEPS_OF, &id.0, |k, _| {
        rows.push((k.to_vec(), codec::u32_at(k, 16)?, codec::u64_at(k, 20)?));
        Ok(true)
    })?;
    for (k, stream, start) in rows {
        w.del(DEPS, &k_dep(stream, start, id))?;
        w.del(DEPS_OF, &k)?;
    }
    Ok(())
}

fn k_aged(stream: u32, start: u64) -> Vec<u8> {
    Out::new().u32(stream).u64(start).done()
}

/// Merges `span` into `stream`'s aged-out spans, as [`cov_insert`] merges coverage, joining spans at most `gap_ns`
/// apart (an age-out's `merge_gap_ns`; 0 joins only spans that overlap or touch).
fn aged_insert(w: &mut W, stream: u32, span: Interval, gap_ns: u64) -> Result<()> {
    let prefix = k32(stream);
    let (mut start, mut end) = (span.start_ns(), span.end_ns());
    if let Some((k, v)) = w.last_below(AGED_OUT, &prefix, &k_aged(stream, start))? {
        let (s, e) = (codec::u64_at(&k, 4)?, codec::u64_at(&v, 0)?);
        if e.saturating_add(gap_ns) >= start {
            start = s;
            end = end.max(e);
        }
    }
    // A span absorbed may reach further and bring the next one within the gap: look again until none does.
    loop {
        let reach = end.saturating_add(gap_ns);
        let hi = reach.checked_add(1).map_or_else(|| prefix_end(&prefix), |e| Some(k_aged(stream, e)));
        let mut absorbed = Vec::new();
        w.scan(AGED_OUT, &k_aged(stream, start), hi.as_deref(), &mut |k, v| {
            absorbed.push((k.to_vec(), codec::u64_at(v, 0)?));
            Ok(true)
        })?;
        let before = end;
        for (k, e) in absorbed {
            w.del(AGED_OUT, &k)?;
            end = end.max(e);
        }
        if end == before {
            break;
        }
    }
    w.put(AGED_OUT, &k_aged(stream, start), &end.to_be_bytes())
}

/// `stream`'s aged-out spans within `window`, clipped to it, the first `max`.
fn aged_within(rd: &impl Reader, stream: u32, window: Interval, max: usize) -> Result<Vec<Interval>> {
    let mut out = Vec::new();
    let push = |s: u64, e: u64, out: &mut Vec<Interval>| {
        if let Some(i) = Interval::new(s.max(window.start_ns()), e.min(window.end_ns())) {
            out.push(i);
        }
    };
    if let Some((k, v)) = rd.last_below(AGED_OUT, &k32(stream), &k_aged(stream, window.start_ns()))? {
        push(codec::u64_at(&k, 4)?, codec::u64_at(&v, 0)?, &mut out);
    }
    let mut rows = Vec::new();
    rd.scan(AGED_OUT, &k_aged(stream, window.start_ns()), Some(&k_aged(stream, window.end_ns())), &mut |k, v| {
        rows.push((codec::u64_at(k, 4)?, codec::u64_at(v, 0)?));
        Ok(rows.len() < max)
    })?;
    for (s, e) in rows {
        push(s, e, &mut out);
    }
    out.truncate(max);
    Ok(out)
}

fn cov_insert(w: &mut W, source: u32, stage: u32, span: Interval) -> Result<()> {
    let prefix = Out::new().u32(source).u32(stage).done();
    let (mut start, mut end) = (span.start_ns(), span.end_ns());
    if let Some((k, v)) = w.last_below(COVERAGE, &prefix, &k_cov(source, stage, start))? {
        let (s, e) = (codec::u64_at(&k, 8)?, codec::u64_at(&v, 0)?);
        if e >= start {
            start = s;
            end = end.max(e);
        }
    }
    let hi = end.checked_add(1).map_or_else(|| prefix_end(&prefix), |e| Some(k_cov(source, stage, e)));
    let mut absorbed = Vec::new();
    w.scan(COVERAGE, &k_cov(source, stage, start), hi.as_deref(), &mut |k, v| {
        absorbed.push((k.to_vec(), codec::u64_at(v, 0)?));
        Ok(true)
    })?;
    for (k, e) in absorbed {
        w.del(COVERAGE, &k)?;
        end = end.max(e);
    }
    w.put(COVERAGE, &k_cov(source, stage, start), &end.to_be_bytes())
}

fn cov_remove(w: &mut W, source: u32, stage: u32, span: Interval) -> Result<()> {
    let prefix = Out::new().u32(source).u32(stage).done();
    let (start, end) = (span.start_ns(), span.end_ns());
    let mut hit = Vec::new();
    if let Some((k, v)) = w.last_below(COVERAGE, &prefix, &k_cov(source, stage, start))? {
        let e = codec::u64_at(&v, 0)?;
        if e > start {
            hit.push((codec::u64_at(&k, 8)?, e));
        }
    }
    w.scan(COVERAGE, &k_cov(source, stage, start), Some(&k_cov(source, stage, end)), &mut |k, v| {
        hit.push((codec::u64_at(k, 8)?, codec::u64_at(v, 0)?));
        Ok(true)
    })?;
    for (s, e) in hit {
        w.del(COVERAGE, &k_cov(source, stage, s))?;
        if s < start {
            w.put(COVERAGE, &k_cov(source, stage, s), &start.to_be_bytes())?;
        }
        if e > end {
            w.put(COVERAGE, &k_cov(source, stage, end), &e.to_be_bytes())?;
        }
    }
    Ok(())
}

/// The stages that have coverage over `source`.
fn stages_of(rd: &impl Reader, source: u32) -> Result<BTreeSet<u32>> {
    let mut out = BTreeSet::new();
    rd.prefix(COVERAGE, &k32(source), |k, _| {
        out.insert(codec::u32_at(k, 4)?);
        Ok(true)
    })?;
    Ok(out)
}

/// Takes `span` out of every stage's coverage over `stream`, saying so in `delta`.
fn uncover(w: &mut W, stream: StreamId, span: Interval, delta: &mut PurgeDelta) -> Result<()> {
    for stage in stages_of(w, stream.0)? {
        cov_remove(w, stream.0, stage, span)?;
        delta.uncovered.push((stream, StageId(stage), span));
    }
    Ok(())
}

// ---- purge jobs ----

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Roots = 0,
    Expanding = 1,
    Removing = 2,
    Cutting = 3,
}

struct Job {
    selector: PurgeSelector,
    phase: Phase,
    /// Range purges: the streams to scan, and how far.
    streams: Vec<u32>,
    pos: u32,
    after: Option<Vec<u8>>,
    next_seq: u64,
    done: u64,
}

impl Job {
    fn encode(&self) -> Vec<u8> {
        let mut o = Out::new().bytes(&codec::selector(&self.selector)).u8(self.phase as u8);
        o = o.u32(u32::try_from(self.streams.len()).unwrap_or(u32::MAX));
        for s in &self.streams {
            o = o.u32(*s);
        }
        o = o.u32(self.pos);
        o = match &self.after {
            Some(a) => o.u8(1).bytes(a),
            None => o.u8(0),
        };
        o.u64(self.next_seq).u64(self.done).done()
    }

    fn decode(b: &[u8]) -> Result<Self> {
        let mut i = In::new(b);
        let selector = codec::read_selector(i.bytes()?)?;
        let phase = match i.u8()? {
            0 => Phase::Roots,
            1 => Phase::Expanding,
            2 => Phase::Removing,
            _ => Phase::Cutting,
        };
        let n = i.u32()?;
        let streams = (0..n).map(|_| i.u32()).collect::<Result<_>>()?;
        let pos = i.u32()?;
        let after = if i.u8()? == 1 { Some(i.bytes()?.to_vec()) } else { None };
        Ok(Self { selector, phase, streams, pos, after, next_seq: i.u64()?, done: i.u64()? })
    }
}

fn load_job(rd: &impl Reader, purge: Uuid) -> Result<Option<Job>> {
    rd.get(PURGE_JOBS, &purge.0)?.map(|b| Job::decode(&b)).transpose()
}

/// Hides `id` for `purge` and queues it to be followed and removed. `late` (an insert, not a step) keeps it for the
/// next step's delta.
fn doom(w: &mut W, purge: Uuid, job: &mut Job, id: Uuid, late: bool, delta: &mut PurgeDelta) -> Result<()> {
    if !job.selector.is_age_out() {
        recording_unknown(w, id)?;
    }
    if doomed(w, id)? || !known(w, id)? {
        return Ok(());
    }
    let Some(entry) = doomed_entry(w, id)? else { return Ok(()) };
    if job.selector.is_age_out() {
        invalidate_aged(w, entry.stream)?;
    } else {
        invalidate(w, entry.stream)?;
    }
    w.put(DOOMED, &id.0, &purge.0)?;
    w.put(PURGE_WORK, &k_pair(purge, id), &[])?;
    let seq = job.next_seq;
    job.next_seq += 1;
    w.put(PURGE_FRONTIER, &k_seq(purge, seq), &id.0)?;
    if late {
        let entry = Doomed { late: true, ..entry };
        w.put(PURGE_LATE, &k_seq(purge, seq), &codec::doomed(&entry))?;
    } else {
        delta.doomed.push(entry);
    }
    Ok(())
}

/// Keeps what an insert hid or cut for a running purge, for that purge's next step's delta.
/// `landed` is the payload being inserted: its entry is marked `late`, since its insert reported it hidden; anything
/// else the insert hid along with it (a chunk's kept frames, say) wasn't, so it isn't.
fn keep_late(w: &mut W, purge: Uuid, job: &mut Job, landed: Uuid, late: PurgeDelta) -> Result<()> {
    for d in late.doomed {
        let d = Doomed { late: d.id == landed, ..d };
        let seq = job.next_seq;
        job.next_seq += 1;
        w.put(PURGE_LATE, &k_seq(purge, seq), &codec::doomed(&d))?;
    }
    for c in late.cut_frames {
        let seq = job.next_seq;
        job.next_seq += 1;
        w.put(PURGE_LATE_CUTS, &k_seq(purge, seq), &codec::cut_frame(&c))?;
    }
    Ok(())
}

/// Marks frame `index` of `chunk` to be cut, and hides what was derived from the chunk over its slot.
fn cut(w: &mut W, purge: Uuid, job: &mut Job, chunk: Uuid, index: u32, delta: &mut PurgeDelta) -> Result<()> {
    recording_unknown(w, chunk)?;
    if w.has(PURGE_CUTS, &k_cut(chunk, purge, index))? {
        return Ok(());
    }
    let Some(mut frames) = chunk_frames(w, chunk)? else { return Ok(()) };
    if frames.get(index as usize).is_none_or(|f| f.cut) {
        return Ok(()); // gone, or another purge cut it first and said so
    }
    let Some((stream, slot)) = frame_slot(w, chunk, index)? else { return Ok(()) };
    invalidate(w, stream)?;
    let value = codec::put_interval(Out::new(), slot).done();
    w.put(PURGE_CUTS, &k_cut(chunk, purge, index), &value)?;
    w.put(PURGE_CUT_ORDER, &k_cut_order(purge, chunk, index), &Out::new().raw(&value).u8(0).done())?;
    // Readers skip it from now on, as they do a hidden payload.
    if let Some(f) = frames.get_mut(index as usize) {
        f.cut = true;
    }
    w.put(CHUNKS, &chunk.0, &codec::frames(&frames))?;
    let tiers = locations(w, chunk)?.iter().map(StorageLocator::tier).collect();
    delta.cut_frames.push(CutFrame { chunk, index, stream, slot, tiers });
    for f in kept_frames(w, chunk)? {
        if frame_row(w, f)?.is_some_and(|(_, i, _)| i == index) {
            doom(w, purge, job, f, false, delta)?;
        }
    }
    for c in children(w, chunk)? {
        if header_of(w, c)?.and_then(|h| h.span()).is_some_and(|s| s.overlaps(slot)) {
            doom(w, purge, job, c, false, delta)?;
        }
    }
    // What depended on the cut frame's stretch of the stream goes too.
    for d in dependents(w, stream.0, slot)? {
        doom(w, purge, job, d, false, delta)?;
    }
    Ok(())
}

/// `age_out`: an age-out's removal, which in place of taking the span out of coverage records it as aged out. Either
/// way the children's `PROV_PARENTS` lists forget it, kept even when empty so a child still reads as derived; their
/// headers still name it.
/// `aged_gap` is `Some(merge_gap_ns)` for an age-out (its span is recorded as aged), `None` for any other purge.
fn remove_one(w: &mut W, purge: Uuid, id: Uuid, aged_gap: Option<u64>, delta: &mut PurgeDelta) -> Result<()> {
    if let Some(entry) = doomed_entry(w, id)? {
        if aged_gap.is_some() {
            invalidate_aged(w, entry.stream)?;
        } else {
            invalidate(w, entry.stream)?;
        }
    }
    w.del(PURGE_WORK, &k_pair(purge, id))?;
    w.del(DOOMED, &id.0)?;
    let kids = children(w, id)?;
    for c in &kids {
        w.del(PROV_CHILDREN, &k_pair(id, *c))?;
        if let Some(b) = w.get(PROV_PARENTS, &c.0)? {
            let rest: Vec<u8> = codec::ids(&b)?.into_iter().filter(|p| *p != id).flat_map(|p| p.0).collect();
            w.put(PROV_PARENTS, &c.0, &rest)?;
        }
    }
    if let Some((chunk, _, subjects)) = frame_row(w, id)? {
        w.del(FRAMES, &id.0)?;
        w.del(CHUNK_FRAMES, &k_pair(chunk, id))?;
        for s in subjects {
            w.del(SUBJECTS, &k_pair(s, id))?;
        }
        w.put(PURGE_REMOVED_FRAMES, &k_pair(purge, id), &[])?;
        return Ok(());
    }
    let Some(h) = header_of(w, id)? else { return Ok(()) };
    w.del(PAYLOADS, &id.0)?;
    // A chunk may have taken this key (a cut first frame's), so only this row's own entry goes.
    let key = k_time(h.stream().0, h.start_ns(), h.seq());
    if w.get(TIME_INDEX, &key)?.as_deref() == Some(&id.0[..]) {
        w.del(TIME_INDEX, &key)?;
    }
    drop_seq(w, h.stream().0, id)?;
    let locs = locations(w, id)?;
    for l in &locs {
        del_location(w, id, l.tier())?;
    }
    w.del(CHUNKS, &id.0)?;
    w.del(CHUNK_PROVENANCE, &id.0)?;
    for p in parents(w, id)? {
        w.del(PROV_CHILDREN, &k_pair(p, id))?;
    }
    w.del(PROV_PARENTS, &id.0)?;
    drop_dependencies(w, id)?;
    for s in h.subjects() {
        w.del(SUBJECTS, &k_pair(s, id))?;
    }
    if let Some(span) = h.span() {
        if let Some(gap) = aged_gap {
            aged_insert(w, h.stream().0, span, gap)?;
        } else {
            uncover(w, h.stream(), span, delta)?;
        }
    }
    // The report is kept for good: an inline copy keeps no bytes in it.
    let removed = RemovedPayload { id, stream: h.stream(), locations: locs.into_iter().map(reported).collect() };
    let count = w.get(PURGE_COUNTS, &purge.0)?.map_or(Ok(0), |b| codec::u64_at(&b, 0))?;
    w.put(PURGE_COUNTS, &purge.0, &(count + 1).to_be_bytes())?;
    w.put(PURGE_REMOVED, &k_pair(purge, id), &codec::removed(&removed))
}

/// One step of a purge, at most about `budget` rows. True when the purge has finished.
fn step(w: &mut W, purge: Uuid, job: &mut Job, budget: usize, delta: &mut PurgeDelta) -> Result<bool> {
    // Rows hidden and frames cut by inserts since the last step.
    for (k, v) in w.rows(PURGE_LATE, &purge.0, usize::MAX)? {
        w.del(PURGE_LATE, &k)?;
        delta.doomed.push(codec::read_doomed(&v)?);
    }
    for (k, v) in w.rows(PURGE_LATE_CUTS, &purge.0, usize::MAX)? {
        w.del(PURGE_LATE_CUTS, &k)?;
        delta.cut_frames.push(codec::read_cut_frame(&v)?);
    }
    match job.phase {
        Phase::Roots => roots(w, purge, job, budget, delta)?,
        _ => {
            let frontier = w.rows(PURGE_FRONTIER, &purge.0, budget)?;
            if !frontier.is_empty() {
                job.phase = Phase::Expanding;
                for (k, v) in frontier {
                    w.del(PURGE_FRONTIER, &k)?;
                    expand(w, purge, job, codec::id_at(&v, 0)?, delta)?;
                    job.done += 1;
                }
                return Ok(false);
            }
            let work = w.rows(PURGE_WORK, &purge.0, budget)?;
            if !work.is_empty() {
                job.phase = Phase::Removing;
                let aged_gap = match job.selector {
                    PurgeSelector::AgeOut { merge_gap_ns, .. } => Some(merge_gap_ns),
                    _ => None,
                };
                for (k, _) in work {
                    remove_one(w, purge, codec::id_at(&k, 16)?, aged_gap, delta)?;
                    job.done += 1;
                }
                return Ok(false);
            }
            job.phase = Phase::Cutting;
            let mut pending = Vec::new();
            w.prefix(PURGE_CUT_ORDER, &purge.0, |k, v| {
                if v.last() == Some(&0) {
                    pending.push((k.to_vec(), v.to_vec()));
                }
                Ok(pending.len() < budget)
            })?;
            if !pending.is_empty() {
                for (k, v) in pending {
                    let chunk = codec::id_at(&k, 16)?;
                    let slot = codec::interval(&mut In::new(&v))?;
                    // The guard in PURGE_CUTS stays until the purge finishes, so a copy hydrated over the slot
                    // meanwhile is still hidden and purged.
                    w.put(
                        PURGE_CUT_ORDER,
                        &k,
                        &Out::new()
                            .raw(
                                v.get(..16)
                                    .ok_or_else(|| CatalogError::Storage("a purge's cut row is corrupt".into()))?,
                            )
                            .u8(1)
                            .done(),
                    )?;
                    if let Some(h) = header_of(w, chunk)? {
                        uncover(w, h.stream(), slot, delta)?;
                    }
                    job.done += 1;
                }
                return Ok(false);
            }
            // Last, a budget at a time: let go of the cut guards (their rows stay, as the report's partial chunks).
            let mut guarded = Vec::new();
            w.prefix(PURGE_CUT_ORDER, &purge.0, |k, v| {
                if v.last() == Some(&1) {
                    guarded.push((k.to_vec(), v.to_vec()));
                }
                Ok(guarded.len() < budget)
            })?;
            if !guarded.is_empty() {
                for (k, v) in guarded {
                    let (chunk, index) = (codec::id_at(&k, 16)?, codec::u32_at(&k, 32)?);
                    w.del(PURGE_CUTS, &k_cut(chunk, purge, index))?;
                    w.put(
                        PURGE_CUT_ORDER,
                        &k,
                        &Out::new()
                            .raw(
                                v.get(..16)
                                    .ok_or_else(|| CatalogError::Storage("a purge's cut row is corrupt".into()))?,
                            )
                            .u8(2)
                            .done(),
                    )?;
                    job.done += 1;
                }
                return Ok(false);
            }
            // Privacy hiding/removal takes precedence over historical ledger work.
            // The pending scope mask remains until this bounded final sweep ends.
            if !recording_range_step(w, purge, job, budget)? {
                return Ok(false);
            }
            finish(w, purge, job, delta)?;
            return Ok(true);
        }
    }
    Ok(false)
}

fn roots(w: &mut W, purge: Uuid, job: &mut Job, budget: usize, delta: &mut PurgeDelta) -> Result<()> {
    let mut left = budget;
    match job.selector.clone() {
        PurgeSelector::Ids(_) => {
            // The ids are in PURGE_IDS (the job keeps none, so a step doesn't rewrite them); `pos` is how many are done.
            let from = u64::from(job.pos);
            let mut rows = Vec::new();
            w.scan(PURGE_IDS, &k_seq(purge, from), prefix_end(&purge.0).as_deref(), &mut |k, v| {
                rows.push((k.to_vec(), codec::id_at(v, 0)?));
                Ok(rows.len() < left)
            })?;
            let mut take = Vec::new();
            for (k, id) in rows {
                w.del(PURGE_IDS, &k)?;
                doom(w, purge, job, id, false, delta)?;
                take.push(id);
            }
            let full = take.len() == left;
            left -= take.len();
            job.pos = u32::try_from(from + take.len() as u64).unwrap_or(u32::MAX);
            if !full {
                job.phase = Phase::Expanding;
            }
        }
        PurgeSelector::AgeOut { committed_by_seq, .. } => {
            // As for ids, from PURGE_IDS; only payloads committed by the cutoff, never an absorbed frame.
            let from = u64::from(job.pos);
            let mut rows = Vec::new();
            w.scan(PURGE_IDS, &k_seq(purge, from), prefix_end(&purge.0).as_deref(), &mut |k, v| {
                rows.push((k.to_vec(), codec::id_at(v, 0)?));
                Ok(rows.len() < left)
            })?;
            let taken = rows.len();
            for (k, id) in rows {
                w.del(PURGE_IDS, &k)?;
                let seq = w.get(PAYLOAD_SEQ, &id.0)?.map(|b| codec::u64_at(&b, 0)).transpose()?;
                if seq.is_some_and(|seq| seq <= committed_by_seq) && w.has(PAYLOADS, &id.0)? {
                    doom(w, purge, job, id, false, delta)?;
                }
            }
            let full = taken == left;
            left -= taken;
            job.pos = u32::try_from(from + taken as u64).unwrap_or(u32::MAX);
            if !full {
                job.phase = Phase::Expanding;
            }
        }
        PurgeSelector::Entity(entity) => {
            let lo = job.after.clone().unwrap_or_else(|| entity.0.to_vec());
            let hi = prefix_end(&entity.0);
            let mut hits = Vec::new();
            w.scan(SUBJECTS, &lo, hi.as_deref(), &mut |k, _| {
                if job.after.as_deref() != Some(k) {
                    hits.push(k.to_vec());
                }
                Ok(hits.len() < left)
            })?;
            w.scan(RECORDING_SUBJECTS, &lo, hi.as_deref(), &mut |k, _| {
                if job.after.as_deref() != Some(k) {
                    hits.push(k.to_vec());
                }
                Ok(hits.len() < left.saturating_mul(2))
            })?;
            hits.sort();
            hits.dedup();
            hits.truncate(left);
            for k in &hits {
                doom(w, purge, job, codec::id_at(k, 16)?, false, delta)?;
            }
            let full = hits.len() == left;
            left -= hits.len();
            if !full {
                job.phase = Phase::Expanding;
                job.after = None;
            } else {
                job.after = hits.last().cloned();
            }
        }
        PurgeSelector::Range { span, .. } => {
            while left > 0 && (job.pos as usize) < job.streams.len() {
                let stream = job.streams[job.pos as usize];
                let reach = w.get(STREAM_LONGEST, &k32(stream))?.map_or(Ok(0), |b| codec::u64_at(&b, 0))?;
                let lo = job.after.clone().unwrap_or_else(|| k_time(stream, span.start_ns().saturating_sub(reach), 0));
                let hi = k_time(stream, span.end_ns(), 0);
                let mut hits = Vec::new();
                w.scan(TIME_INDEX, &lo, Some(&hi), &mut |k, v| {
                    if job.after.as_deref() != Some(k) {
                        hits.push((k.to_vec(), codec::id_at(v, 0)?));
                    }
                    Ok(hits.len() < left)
                })?;
                for (_, id) in &hits {
                    root_in_range(w, purge, job, *id, span, delta)?;
                }
                left -= hits.len().min(left);
                if left > 0 {
                    job.pos += 1;
                    job.after = None;
                } else {
                    job.after = hits.last().map(|(k, _)| k.clone());
                }
            }
            if job.pos as usize >= job.streams.len() {
                job.phase = Phase::Expanding;
            }
        }
    }
    job.done += u64::try_from(budget - left).unwrap_or(0);
    Ok(())
}

fn root_in_range(
    w: &mut W,
    purge: Uuid,
    job: &mut Job,
    id: Uuid,
    span: Interval,
    delta: &mut PurgeDelta,
) -> Result<()> {
    let Some(h) = header_of(w, id)? else { return Ok(()) };
    let Some(own) = h.span() else { return Ok(()) };
    if !own.overlaps(span) {
        return Ok(());
    }
    let whole = own.start_ns() >= span.start_ns() && own.end_ns() <= span.end_ns();
    match chunk_frames(w, id)? {
        Some(frames) if !whole => {
            // A frame is cut when the purge overlaps its slot, the rule a raw frame's span follows.
            let slots = frame_slots(own, &frames);
            for (i, f) in frames.iter().enumerate() {
                if slots.get(i).is_some_and(|slot| slot.overlaps(span)) && !f.cut {
                    cut(w, purge, job, id, u32::try_from(i).unwrap_or(u32::MAX), delta)?;
                }
            }
            Ok(())
        }
        _ => doom(w, purge, job, id, false, delta),
    }
}

/// Follows a hidden id to what was derived from it.
fn expand(w: &mut W, purge: Uuid, job: &mut Job, id: Uuid, delta: &mut PurgeDelta) -> Result<()> {
    if job.selector.is_age_out() {
        // Nothing derived from it goes. A whole chunk takes its kept frames' rows (they are its bytes), and what was
        // derived from those frames stays.
        for f in kept_frames(w, id)? {
            doom(w, purge, job, f, false, delta)?;
        }
        return Ok(());
    }
    for c in children(w, id)? {
        doom(w, purge, job, c, false, delta)?;
    }
    // What depended on its stretch of its stream goes too. An absorbed frame's are found through its cut.
    if let Some(h) = header_of(w, id)?
        && let Some(span) = h.span()
    {
        for d in dependents(w, h.stream().0, span)? {
            doom(w, purge, job, d, false, delta)?;
        }
    }
    for f in kept_frames(w, id)? {
        doom(w, purge, job, f, false, delta)?;
    }
    if let Some((chunk, index, _)) = frame_row(w, id)?
        && !doomed(w, chunk)?
    {
        cut(w, purge, job, chunk, index, delta)?;
    }
    Ok(())
}

/// Ends a purge whose rows are gone: takes a range out of the purged streams' coverage, drops the job, and marks
/// the record finished. The report stays in PURGE_REMOVED, PURGE_REMOVED_FRAMES, and PURGE_CUT_ORDER, read back by
/// [`report_of`], so this step does no work in proportion to the purge's size.
fn finish(w: &mut W, purge: Uuid, job: &Job, delta: &mut PurgeDelta) -> Result<()> {
    if let PurgeSelector::Range { span, .. } = &job.selector {
        for &s in &job.streams {
            uncover(w, StreamId(s), *span, delta)?;
        }
    }
    w.del(PURGE_JOBS, &purge.0)?;
    if let Some(b) = w.get(PURGES, &purge.0)? {
        let mut record = codec::read_record(&b)?;
        record.report = Some(PurgeReport {
            purge_id: purge,
            removed: Vec::new(),
            removed_frames: Vec::new(),
            partial_chunks: Vec::new(),
        });
        w.put(PURGES, &purge.0, &codec::record(&record))?;
    }
    bump_purge_gen(w)?;
    w.put(META, b"scrub_due", &[1])
}

/// Counts purges finished and scrubs requested, so a scrub can tell one happened while it copied.
fn bump_purge_gen(w: &mut W) -> Result<()> {
    let gen_now = w.get(META, b"purge_gen")?.map_or(Ok(0), |b| codec::u64_at(&b, 0))?;
    w.put(META, b"purge_gen", &(gen_now + 1).to_be_bytes())
}

/// `META` "stream_generation": one per stream registered since the key arrived (0 before the first).
fn stream_generation(rd: &impl Reader) -> Result<u64> {
    rd.get(META, b"stream_generation")?.map_or(Ok(0), |b| codec::u64_at(&b, 0))
}

fn swap_count(rd: &impl Reader) -> Result<u64> {
    rd.get(META, b"swap_count")?.map_or(Ok(0), |b| codec::u64_at(&b, 0))
}

fn purge_gen(rd: &impl Reader) -> Result<u64> {
    rd.get(META, b"purge_gen")?.map_or(Ok(0), |b| codec::u64_at(&b, 0))
}

/// The keepers' names (`META` "keepers": a count, then the names), sorted.
fn keepers_of(rd: &impl Reader) -> Result<Vec<String>> {
    let Some(b) = rd.get(META, b"keepers")? else { return Ok(Vec::new()) };
    let mut i = In::new(&b);
    let mut names = Vec::new();
    for _ in 0..i.u32()? {
        names.push(i.string()?);
    }
    Ok(names)
}

fn put_keepers(w: &mut W, names: &[String]) -> Result<()> {
    let mut o = Out::new().u32(u32::try_from(names.len()).unwrap_or(u32::MAX));
    for n in names {
        o = o.str(n);
    }
    w.put(META, b"keepers", &o.done())
}

/// A finished purge's report, read from the rows its steps left (a read, so it never holds the writer).
fn report_of(rd: &impl Reader, purge: Uuid) -> Result<PurgeReport> {
    let mut removed = Vec::new();
    rd.prefix(PURGE_REMOVED, &purge.0, |_, v| {
        removed.push(codec::read_removed(v)?);
        Ok(true)
    })?;
    let mut removed_frames = Vec::new();
    rd.prefix(PURGE_REMOVED_FRAMES, &purge.0, |k, _| {
        removed_frames.push(codec::id_at(k, 16)?);
        Ok(true)
    })?;
    let mut cuts = Vec::new();
    rd.prefix(PURGE_CUT_ORDER, &purge.0, |k, _| {
        cuts.push((codec::id_at(k, 16)?, codec::u32_at(k, 32)?));
        Ok(true)
    })?;
    let mut partial_chunks: Vec<PartialChunk> = Vec::new();
    for (chunk, index) in cuts {
        let Some(h) = header_of(rd, chunk)? else { continue };
        match partial_chunks.last_mut() {
            Some(p) if p.chunk == chunk => p.frames.push(index),
            _ => partial_chunks.push(PartialChunk {
                chunk,
                stream: h.stream(),
                frames: vec![index],
                locations: locations(rd, chunk)?,
            }),
        }
    }
    Ok(PurgeReport { purge_id: purge, removed, removed_frames, partial_chunks })
}

/// Whether a purge removed `id` (a payload or an absorbed frame). The per-purge rows are written as the purge goes,
/// so a frame gone mid-purge, before its report, counts too: a point lookup per purge, and purges are few.
fn purged(rd: &impl Reader, id: Uuid) -> Result<bool> {
    let mut ids = Vec::new();
    rd.scan(PURGE_ORDER, &[], None, &mut |_, v| {
        ids.push(codec::id_at(v, 0)?);
        Ok(true)
    })?;
    for purge in ids {
        let key = k_pair(purge, id);
        if rd.has(PURGE_REMOVED, &key)? || rd.has(PURGE_REMOVED_FRAMES, &key)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// A purge's record, with its report read back when it has finished.
fn record_of(rd: &impl Reader, purge: Uuid) -> Result<Option<PurgeRecord>> {
    let Some(b) = rd.get(PURGES, &purge.0)? else { return Ok(None) };
    let mut record = codec::read_record(&b)?;
    if record.report.is_some() {
        record.report = Some(report_of(rd, purge)?);
    }
    Ok(Some(record))
}

// ---- scrub ----

/// A written key: its table's index in `ALL`, and the key.
type Touched = (usize, Vec<u8>);

/// A scrub in progress: the fresh file and how far the copy has got.
struct Scrub {
    tmp: PathBuf,
    db: Database,
    /// The purge count when it began: a purge that finishes later may leave bytes in the fresh file's freed pages,
    /// so the swap then leaves another scrub due.
    started_gen: u64,
    table: usize,
    after: Option<Vec<u8>>,
    copied: u64,
    /// The live bytes written to the fresh file so far (copies and carries, outside `META`).
    bytes: i64,
    /// Whether the finished copy was flushed to disk, so the swap's own commit has only its few keys to flush.
    flushed: bool,
}

/// The test-only crash points of a scrub: with `CONTINUUM_SCRUB_CRASH_AT` set to a point's name, the process exits
/// there at once, with no destructor run and no file closed, as if it had been killed.
#[cfg(test)]
fn crash_point(name: &str) {
    if std::env::var("CONTINUUM_SCRUB_CRASH_AT").is_ok_and(|at| at == name) {
        std::process::exit(86);
    }
}

#[cfg(not(test))]
#[inline(always)]
fn crash_point(_: &str) {}

/// Adds `w`'s change in live bytes to `META` "live_bytes", once a scrub has counted them (before that the key is
/// absent and there is nothing to keep up).
fn note_live_bytes(w: &mut W) -> Result<()> {
    if w.bytes == 0 {
        return Ok(());
    }
    let Some(b) = w.get(META, b"live_bytes")? else { return Ok(()) };
    let now = codec::u64_at(&b, 0)?.saturating_add_signed(w.bytes);
    w.put(META, b"live_bytes", &now.to_be_bytes())
}

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
}

fn io_error(e: std::io::Error) -> CatalogError {
    CatalogError::Storage(Box::new(e))
}

/// What every call answers once [`Catalog::close`] has closed the file.
fn closed() -> CatalogError {
    CatalogError::Storage("the catalog is closed (the broker is stopping)".into())
}

/// The bytes a file's blocks take on disk.
fn disk_bytes(meta: &std::fs::Metadata) -> u64 {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        meta.blocks().saturating_mul(512)
    }
    #[cfg(not(unix))]
    {
        meta.len()
    }
}

/// Copies `keys` of table `t` from `from` to `to` as they are now in `from` (deleting what's gone).
fn carry(from: &R, to: &mut W, keys: &[(usize, Vec<u8>)]) -> Result<()> {
    for (t, k) in keys {
        let Some(&def) = ALL.get(*t) else { continue };
        match from.get(def, k)? {
            Some(v) => to.put(def, k, &v)?,
            None => {
                to.del(def, k)?;
            }
        }
    }
    Ok(())
}

/// Flushes the directory holding `path`, so a rename into it survives a power loss.
fn sync_dir(path: &Path) -> Result<()> {
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    std::fs::File::open(dir).and_then(|d| d.sync_all()).map_err(|e| CatalogError::Storage(Box::new(e)))
}

/// Process-local measurements of a successful catalog open. Never persisted in the catalog.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpenTimings {
    /// Recovering an interrupted scrub.
    pub scrub_recovery: std::time::Duration,
    /// Opening/creating the redb database, including redb recovery when needed.
    pub database_open: std::time::Duration,
    /// Checking the catalog format, or creating a new file at it.
    pub schema_setup: std::time::Duration,
    /// Repairing current-era recording certification.
    pub recording_repair: std::time::Duration,
    /// Elapsed open time through successful initialization, including unclassified overhead.
    pub total: std::time::Duration,
}

/// The catalog in a redb file.
pub struct RedbCatalog {
    path: PathBuf,
    /// The live file. Readers take it for a moment to open a snapshot; writers hold it shared for their whole
    /// transaction; only a scrub's swap and [`Catalog::close`] hold it alone. `None` once closed.
    db: RwLock<Option<Arc<Database>>>,
    /// The scrub, if one is running. Writers never take this lock (lock order: scrub, then db, then touched).
    scrub: Mutex<Option<Scrub>>,
    /// While a scrub runs: the keys written since it began, to carry over to its fresh file.
    scrubbing: AtomicBool,
    touched: Mutex<BTreeSet<(usize, Vec<u8>)>>,
    /// How long the last swap held the catalog alone.
    last_swap: Mutex<Option<(std::time::Duration, usize)>>,
    windows: Mutex<WindowCache>,
    window_snapshots: Mutex<u64>,
    /// The streams whose recording history `open` found marked unknown with no row left to say why, and restored.
    recording_restored: Vec<u32>,
    /// The native streams still marked unknown after `open`, each with why the repair held it back.
    recording_held: Vec<(u32, String)>,
    open_timings: OpenTimings,
    /// When the last durable (fsynced) commit was, and whether commits since are waiting for one.
    durable: Mutex<std::time::Instant>,
    undurable: AtomicBool,
}

/// How long ordinary commits go without an fsync. Each commit is atomic and consistent either way; a crash loses at most
/// the commits of about this long. Fsyncing every writer tick was about 170 fsyncs a second on the Linux machine, most
/// of the disk's busy time.
pub const DURABLE_EVERY: std::time::Duration = std::time::Duration::from_secs(1);

impl RedbCatalog {
    /// Process-local phase timings from this successful open. No catalog reads are performed.
    #[must_use]
    pub fn open_timings(&self) -> OpenTimings {
        self.open_timings
    }

    /// The streams whose current-era recording history this open restored, for the broker to log.
    #[must_use]
    pub fn recording_restored(&self) -> &[u32] {
        &self.recording_restored
    }

    /// The instant audio originals that still hold their stream's history unknown, each with the chunk and
    /// frame index that hold its samples once it has been flushed. Read-only.
    ///
    /// # Errors
    /// A storage error.
    pub fn instant_originals(&self) -> Result<Vec<InstantOriginal>> {
        self.read(|r| {
            let mut unknown = Vec::new();
            r.scan(RECORDING_META, &[], None, &mut |k, v| {
                if !crate::recording::Meta::decode(v)?.current_known {
                    unknown.push(codec::u32_at(k, 0)?);
                }
                Ok(true)
            })?;
            let mut out = Vec::new();
            for stream in unknown {
                let Some(info) = r.get(STREAMS, &k32(stream))?.map(|b| codec::read_stream_info(&b)).transpose()? else {
                    continue;
                };
                if !crate::recording::native(&info) || !info.name.ends_with(".audio") {
                    continue;
                }
                let mut rows = Vec::new();
                r.prefix(RECORDING_ROWS, &Out::new().u32(stream).done(), |_, v| {
                    let row = crate::recording::Row::decode(v)?;
                    if row.subjects_complete && recording_reason(&row, true) {
                        rows.push((row.input.chunk_id, row.input.start_ns));
                    }
                    Ok(true)
                })?;
                for (original, start_ns) in rows {
                    let frame = frame_row(r, original)?.map(|(chunk, index, _)| (chunk, index));
                    out.push(InstantOriginal { stream: StreamId(stream), original, start_ns, frame });
                }
            }
            Ok(out)
        })
    }

    /// Gives an instant audio original the end its own samples bound: its row stays unknown, now over
    /// `[start, end_ns)` instead of no extent, so it no longer holds the whole stream; then the stream is restored if
    /// nothing else holds it. The caller derives `end_ns` from the original's stored samples. Returns whether the
    /// stream's history is known after.
    ///
    /// # Errors
    /// [`CatalogError::NotFound`] for an unknown original, [`CatalogError::Invalid`] for one that isn't an instant
    /// audio original or an end not after its start.
    pub fn bound_instant_original(&self, original: Uuid, end_ns: u64) -> Result<bool> {
        self.write(|w| {
            let key = w
                .get(RECORDING_IDS, &original.0)?
                .ok_or_else(|| CatalogError::NotFound(format!("recording original {original}")))?;
            let stream = codec::u32_at(&key, 0)?;
            let info = w
                .get(STREAMS, &k32(stream))?
                .map(|b| codec::read_stream_info(&b))
                .transpose()?
                .ok_or_else(|| CatalogError::NotFound(format!("stream {stream}")))?;
            let mut row = crate::recording::Row::decode(
                &w.get(RECORDING_ROWS, &key)?.ok_or_else(|| crate::recording::invalid("missing recording row"))?,
            )?;
            let audio = crate::recording::native(&info) && info.name.ends_with(".audio");
            if !audio || !row.subjects_complete || !recording_reason(&row, true) {
                return Err(crate::recording::invalid("not an instant audio original"));
            }
            let span = Interval::new(row.input.start_ns, end_ns)
                .ok_or_else(|| crate::recording::invalid("a bounded end must be after the original's start"))?;
            row.end_ns = end_ns;
            row.input.spans = vec![span];
            w.put(RECORDING_ROWS, &key, &row.encode())?;
            let mut meta = recording_meta(w, StreamId(stream))?;
            meta.longest = meta.longest.max(row.reach());
            recording_save_meta(w, StreamId(stream), &meta)?;
            bump(w, StreamId(stream))?;
            let swept = recording_swept(w)?;
            Ok(matches!(recording_restore_one(w, stream, &swept)?, Restore::Restored))
        })
    }

    /// The native streams this open left marked unknown, each with why, for the broker to log.
    #[must_use]
    pub fn recording_held(&self) -> &[(u32, String)] {
        &self.recording_held
    }

    fn purge_work(
        &self,
        purge_id: Uuid,
        budget: usize,
        mut should_yield: Option<&mut dyn FnMut() -> bool>,
    ) -> Result<PurgeProgress> {
        if budget == 0 {
            return Err(CatalogError::Invalid("a purge step's budget must be at least 1".into()));
        }
        // Ok(progress) while running; Err(delta) once finished (this step's delta, or empty when it ended before).
        let stepped = self.write(|w| {
            let Some(b) = w.get(PURGES, &purge_id.0)? else {
                return Err(CatalogError::NotFound(format!("purge {purge_id}")));
            };
            if codec::read_record(&b)?.report.is_some() {
                return Ok(Err(PurgeDelta::default()));
            }
            let mut job =
                load_job(w, purge_id)?.ok_or_else(|| CatalogError::NotFound(format!("purge {purge_id} has no job")))?;
            let mut delta = PurgeDelta::default();
            let mut remaining = budget;
            loop {
                let chunk = if should_yield.is_some() { remaining.min(PURGE_BATCH_ROWS) } else { remaining };
                let before = job.done;
                if step(w, purge_id, &mut job, chunk, &mut delta)? {
                    return Ok(Err(delta));
                }
                // A root/phase with no matching rows still advances a cursor and must not create an unbounded loop.
                let used = usize::try_from(job.done.saturating_sub(before)).unwrap_or(remaining).max(1);
                remaining = remaining.saturating_sub(used);
                if let Some(yield_now) = should_yield.as_deref_mut()
                    && remaining > 0
                    && !yield_now()
                {
                    continue;
                }
                break;
            }
            w.put(PURGE_JOBS, &purge_id.0, &job.encode())?;
            let phase = match job.phase {
                Phase::Roots | Phase::Expanding => PurgePhase::Collecting,
                Phase::Removing => PurgePhase::Removing,
                Phase::Cutting => PurgePhase::Finishing,
            };
            Ok(Ok(PurgeProgress::Running { phase, done: job.done, delta }))
        })?;
        match stepped {
            Ok(progress) => Ok(progress),
            // Read the report after the write transaction; report construction still occupies the broker writer thread.
            Err(delta) => Ok(PurgeProgress::Removed { report: self.read(|r| report_of(r, purge_id))?, delta }),
        }
    }

    /// Opens the catalog at `path`, creating it when missing. A scrub cut short by a restart is thrown away (its
    /// fresh file deleted) and starts over.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] when the file can't be opened or created.
    ///
    /// An open that redb refuses as corrupt leaves the marker `path.open-failed`, saying why, and every later open
    /// refuses at once while it is there, without touching the file: each attempt otherwise runs redb's repair again,
    /// which rewrites the file's header, so a supervisor restarting the broker would keep changing the evidence (and
    /// possibly the file) every few seconds. Remove the marker to try again, after keeping a copy of the file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let opened_at = std::time::Instant::now();
        let path = path.as_ref().to_path_buf();
        let failed = Self::open_failed_path(&path);
        if let Ok(why) = std::fs::read_to_string(&failed) {
            return Err(CatalogError::Storage(
                format!(
                    "not opening {}: an earlier open found it corrupt ({}), so it is left as it is; keep a copy of \
                     it, then remove {} to try again",
                    path.display(),
                    why.trim(),
                    failed.display()
                )
                .into(),
            ));
        }
        let phase_started = std::time::Instant::now();
        Self::recover_scrub(&path)?;
        let scrub_recovery = phase_started.elapsed();
        let phase_started = std::time::Instant::now();
        let db = match Database::create(&path) {
            Ok(db) => db,
            Err(e) => {
                if matches!(e, redb::DatabaseError::Storage(redb::StorageError::Corrupted(_))) {
                    let why = format!("{e}, at {} ns since the epoch", now_ns());
                    if std::fs::write(&failed, &why).is_ok() {
                        let _ = sync_dir(&path);
                    }
                }
                return Err(e.into());
            }
        };
        let database_open = phase_started.elapsed();
        let phase_started = std::time::Instant::now();
        Self::check_format(&db)?;
        let schema_setup = phase_started.elapsed();
        let phase_started = std::time::Instant::now();
        let (recording_restored, recording_held) = Self::repair_recording_known(&db)?;
        let recording_repair = phase_started.elapsed();
        Ok(Self {
            durable: Mutex::new(std::time::Instant::now()),
            undurable: AtomicBool::new(false),
            open_timings: OpenTimings {
                scrub_recovery,
                database_open,
                schema_setup,
                recording_repair,
                total: opened_at.elapsed(),
            },
            recording_restored,
            recording_held,
            path,
            db: RwLock::new(Some(Arc::new(db))),
            scrub: Mutex::new(None),
            scrubbing: AtomicBool::new(false),
            touched: Mutex::new(BTreeSet::new()),
            last_swap: Mutex::new(None),
            windows: Mutex::new(WindowCache::default()),
            window_snapshots: Mutex::new(0),
        })
    }

    /// `recording_repair_known` in its own write.
    fn repair_recording_known(db: &Database) -> Result<RecordingRepair> {
        let mut w = W::new(db.begin_write()?);
        let (restored, held) = recording_repair_known(&mut w)?;
        if restored.is_empty() {
            return Ok((restored, held));
        }
        crate::restart::changed(&mut w)?;
        // Includes any opaque restart projection retired by the revision advance above.
        note_live_bytes(&mut w)?;
        w.txn.commit()?;
        Ok((restored, held))
    }

    /// Creates a new, empty file at [`FORMAT`] (every table, and the META values a fresh file starts with), and
    /// refuses any other file: one stamped with another format, or unstamped with streams in it.
    fn check_format(db: &Database) -> Result<()> {
        use redb::TableHandle;
        let stamped = {
            let txn = db.begin_read()?;
            let names: BTreeSet<String> = txn.list_tables()?.map(|table| table.name().to_owned()).collect();
            let r = R::new(txn);
            let stamped = if names.contains(META.name()) {
                r.get(META, b"format")?.map(|b| codec::u64_at(&b, 0)).transpose()?
            } else {
                None
            };
            if stamped.is_none() && names.contains(STREAMS.name()) && !r.rows(STREAMS, &[], 1)?.is_empty() {
                return Err(CatalogError::Storage(
                    format!(
                        "the catalog file has no format; this code reads only format {FORMAT}: create it again \
                         (pack-reset-catalog)"
                    )
                    .into(),
                ));
            }
            if stamped == Some(FORMAT) {
                // What creation made: a missing table or value is corruption, never empty history to start over.
                if let Some(missing) = ALL.iter().find(|def| !names.contains(def.name())) {
                    return Err(CatalogError::Storage(
                        format!("the catalog file is missing its {} table", missing.name()).into(),
                    ));
                }
                recording_era(&r)?;
                if crate::restart::stamp(&r)?.is_none() || !crate::restart::ready(&r)? {
                    return Err(CatalogError::Storage("the catalog is missing its restart tracking".into()));
                }
            }
            stamped
        };
        match stamped {
            Some(FORMAT) => Ok(()),
            Some(other) => Err(CatalogError::Storage(
                format!(
                    "the catalog file is format {other}; this code reads only format {FORMAT}: create it again \
                     (pack-reset-catalog)"
                )
                .into(),
            )),
            None => {
                let mut w = W::new(db.begin_write()?);
                for &def in ALL {
                    w.txn.open_table(def)?;
                }
                w.put(META, b"format", &FORMAT.to_be_bytes())?;
                w.put(META, b"recording_era_start_seq", &RECORDING_ERA_START.to_be_bytes())?;
                crate::restart::create(&mut w)?;
                w.txn.commit()?;
                Ok(())
            }
        }
    }

    /// Settles what a scrub cut short left beside `path`. The swap writes the marker `path.scrub-done` (flushed) only
    /// once the fresh file's last commit is durable, renames the fresh file over the catalog, flushes the directory,
    /// and then removes the marker. So: a fresh file with the marker finished its swap but the rename may not have
    /// reached the disk, and it is put in place; a fresh file without the marker never got to its swap and is thrown
    /// away (without opening it, so a crash mid-copy costs no repair of a large half-written file); a marker alone is
    /// left from a swap that completed.
    fn recover_scrub(path: &Path) -> Result<()> {
        let (scrub, done) = (Self::scrub_path(path), Self::done_path(path));
        match (scrub.exists(), done.exists()) {
            (true, true) => {
                std::fs::rename(&scrub, path).map_err(io_error)?;
                sync_dir(path)?;
                std::fs::remove_file(&done).map_err(io_error)?;
            }
            (true, false) => std::fs::remove_file(&scrub).map_err(io_error)?,
            (false, true) => std::fs::remove_file(&done).map_err(io_error)?,
            (false, false) => return Ok(()),
        }
        sync_dir(path)
    }

    /// The marker a corrupt open leaves (see [`RedbCatalog::open`]).
    fn open_failed_path(path: &Path) -> PathBuf {
        let mut p = path.as_os_str().to_owned();
        p.push(".open-failed");
        PathBuf::from(p)
    }

    fn done_path(path: &Path) -> PathBuf {
        let mut p = path.as_os_str().to_owned();
        p.push(".scrub-done");
        PathBuf::from(p)
    }

    fn scrub_path(path: &Path) -> PathBuf {
        let mut p = path.as_os_str().to_owned();
        p.push(".scrub");
        PathBuf::from(p)
    }

    /// While the copy runs: forgets the written keys it has yet to reach (a later copy step reads them as they are
    /// then) and carries up to `budget` of the ones behind it, so the set of written keys stays about as large as
    /// one step's writes rather than growing for the whole copy.
    fn drain_touched(&self, s: &mut Scrub, budget: usize) -> Result<()> {
        let (behind, ahead): (Vec<Touched>, BTreeSet<Touched>) = {
            let mut touched = self.touched.lock().unwrap_or_else(PoisonError::into_inner);
            let ahead = if s.table < ALL.len() {
                // The smallest key after the copy's last one (that key with a 0 byte added), or the start of the
                // table when it hasn't begun: from there on, the rest of this table and every table after it.
                let from = s.after.clone().map_or_else(Vec::new, |mut k| {
                    k.push(0);
                    k
                });
                touched.split_off(&(s.table, from))
            } else {
                BTreeSet::new()
            };
            (std::iter::from_fn(|| touched.pop_first()).take(budget).collect(), ahead)
        };
        // Freed after the writers' lock is let go.
        drop(ahead);
        if behind.is_empty() {
            return Ok(());
        }
        let db = self.live()?;
        let from = R::new(db.begin_read()?);
        let mut w = W::new(s.db.begin_write()?);
        w.txn.set_durability(Durability::None)?;
        carry(&from, &mut w, &behind)?;
        w.txn.commit()?;
        s.bytes += w.bytes;
        s.copied += behind.len() as u64;
        Ok(())
    }

    /// The live database, for a call that holds it only to open a snapshot.
    fn live(&self) -> Result<Arc<Database>> {
        self.db.read().unwrap_or_else(PoisonError::into_inner).as_ref().map(Arc::clone).ok_or_else(closed)
    }

    /// Each table's stored and metadata bytes (redb's stats), by table name: what the catalog's contents take before
    /// redb's free pages and file slack. For measuring growth, such as what an aged hour leaves behind.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] when the file can't be read.
    pub fn table_bytes(&self) -> Result<Vec<(String, u64)>> {
        use redb::{ReadableTableMetadata, TableHandle};
        self.read(|r| {
            let mut out = Vec::new();
            for def in ALL {
                match r.txn.open_table(*def) {
                    Ok(t) => {
                        let s = t.stats()?;
                        out.push((def.name().to_owned(), s.stored_bytes() + s.metadata_bytes()));
                    }
                    Err(redb::TableError::TableDoesNotExist(_)) => {}
                    Err(e) => return Err(e.into()),
                }
            }
            Ok(out)
        })
    }

    fn read<T>(&self, f: impl FnOnce(&R) -> Result<T>) -> Result<T> {
        let db = self.live()?;
        let r = R::new(db.begin_read()?);
        f(&r)
    }

    fn write<T>(&self, f: impl FnOnce(&mut W) -> Result<T>) -> Result<T> {
        self.write_restart(true, f)
    }

    fn write_restart<T>(&self, advance: bool, f: impl FnOnce(&mut W) -> Result<T>) -> Result<T> {
        // Held for the whole write, so a scrub's swap (or the close) waits for it and it never writes to a file being
        // replaced.
        let guard = self.db.read().unwrap_or_else(PoisonError::into_inner);
        let db = guard.as_ref().ok_or_else(closed)?;
        let mut w = W::new(db.begin_write()?);
        let out = f(&mut w)?;
        if advance {
            crate::restart::changed(&mut w)?;
        }
        note_live_bytes(&mut w)?;
        let touched = std::mem::take(&mut w.touched);
        // Durable at most every DURABLE_EVERY: the commits between are made durable by the next durable one, by
        // `flush_durable` (the broker's writer calls it when idle), or by `close`.
        let mut durable = self.durable.lock().unwrap_or_else(PoisonError::into_inner);
        let fsync = durable.elapsed() >= DURABLE_EVERY;
        if !fsync {
            w.txn.set_durability(Durability::None)?;
        }
        w.txn.commit()?;
        if fsync {
            *durable = std::time::Instant::now();
            self.undurable.store(false, Ordering::Release);
        } else {
            self.undurable.store(true, Ordering::Release);
        }
        drop(durable);
        // Still holding the live file shared, so a swap can't happen before these keys are noted.
        if self.scrubbing.load(Ordering::Acquire) {
            self.touched.lock().unwrap_or_else(PoisonError::into_inner).extend(touched);
        }
        drop(guard);
        Ok(out)
    }

    /// How long the last scrub's swap held the catalog alone, blocking every other call (a measurement, for tests and
    /// the broker's metrics).
    #[must_use]
    pub fn last_swap_hold(&self) -> Option<std::time::Duration> {
        self.last_swap.lock().unwrap_or_else(PoisonError::into_inner).map(|(hold, _)| hold)
    }

    /// How many written keys the last scrub's swap carried while it held the catalog alone: at most [`SWAP_DELTA`],
    /// whatever the catalog holds, since the catch-up steps carry the rest outside the lock.
    #[must_use]
    pub fn last_swap_carried(&self) -> Option<usize> {
        self.last_swap.lock().unwrap_or_else(PoisonError::into_inner).map(|(_, carried)| carried)
    }

    /// The file it lives in.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Makes every commit so far durable with one fsync, when some aren't yet (see [`DURABLE_EVERY`]).
    fn flush(&self) -> Result<()> {
        if !self.undurable.load(Ordering::Acquire) {
            return Ok(());
        }
        let guard = self.db.read().unwrap_or_else(PoisonError::into_inner);
        let Some(db) = guard.as_ref() else { return Ok(()) };
        let mut durable = self.durable.lock().unwrap_or_else(PoisonError::into_inner);
        // An empty durable commit persists the non-durable ones before it.
        db.begin_write()?.commit()?;
        *durable = std::time::Instant::now();
        self.undurable.store(false, Ordering::Release);
        Ok(())
    }

    /// Indexes `payloads` in `w` (see [`Catalog::insert`]). A repeat must come with the dependencies it was indexed
    /// with (`given`, sorted (stream, start, end) by id; none when absent): the same header with different purge
    /// coverage is a Conflict, never an answer that it's there already.
    fn insert_rows(
        w: &mut W,
        payloads: Vec<NewPayload>,
        given: &std::collections::HashMap<Uuid, Vec<(u32, u64, u64)>>,
    ) -> Result<Vec<Inserted>> {
        // Already indexed with identical header bytes (or earlier in this batch): its commit sequence, unwritten.
        // A frame a chunk absorbed since, sent again with the stream and start its frame index holds, is a repeat
        // too, answering the chunk's sequence: the frame's own header is gone, the chunk's stands for it.
        // Each entry is (the payload's id, whether it is a repeat, the payload whose sequence answers).
        let mut fresh: Vec<NewPayload> = Vec::new();
        let mut first: std::collections::HashMap<Uuid, usize> = std::collections::HashMap::new();
        let mut plan: Vec<(Uuid, bool, Uuid)> = Vec::new();
        for p in payloads {
            let id = p.header.id();
            if w.get(PAYLOADS, &id.0)?.as_deref() == Some(p.header.as_bytes()) {
                if dependencies_of(w, id)? != given.get(&id).cloned().unwrap_or_default() {
                    return Err(CatalogError::Conflict(format!(
                        "{id} is indexed already with other dependencies: a repeat must name the same ones"
                    )));
                }
                plan.push((id, true, id));
            } else if first.get(&id).is_some_and(|&i| fresh[i].header == p.header) {
                plan.push((id, true, id));
            } else if let Some(chunk) = absorbed_as(w, &p.header)? {
                plan.push((id, true, chunk));
            } else {
                first.insert(id, fresh.len());
                plan.push((id, false, id));
                fresh.push(p);
            }
        }
        Self::check(w, &fresh)?;
        for p in fresh {
            Self::put(w, p)?;
        }
        plan.into_iter().map(|(id, duplicate, answers)| inserted_as(w, id, answers, duplicate)).collect()
    }

    fn check(w: &W, batch: &[NewPayload]) -> Result<()> {
        let mut ids = BTreeSet::new();
        let mut keys = BTreeSet::new();
        for p in batch {
            let h = &p.header;
            let id = h.id();
            if !w.has(STREAMS, &k32(h.stream().0))? {
                return Err(CatalogError::NotFound(format!("stream {} of payload {id}", h.stream().0)));
            }
            span_of(h)?;
            if recording_get(w, id)?.is_some() {
                return Err(CatalogError::Conflict(format!("original recording identity {id} is reserved")));
            }
            if known(w, id)? || !ids.insert(id) {
                return Err(CatalogError::Conflict(format!("payload {id} is indexed already, with another header")));
            }
            if matches!(&p.location, Some(StorageLocator::Inline { bytes }) if bytes.len() > continuum_core::MAX_INLINE_BYTES)
            {
                return Err(CatalogError::Invalid(format!(
                    "payload {id}'s inline bytes are over {} bytes: put it in shared memory",
                    continuum_core::MAX_INLINE_BYTES
                )));
            }
            let key = k_time(h.stream().0, h.start_ns(), h.seq());
            if w.has(TIME_INDEX, &key)? || !keys.insert(key) {
                return Err(CatalogError::Conflict(format!(
                    "stream {} has a payload at {} seq {}",
                    h.stream().0,
                    h.start_ns(),
                    h.seq()
                )));
            }
            let ps = h.parents();
            if ps.contains(&id) {
                return Err(CatalogError::Invalid(format!("payload {id} names itself as a parent")));
            }
            for q in ps {
                if !ids.contains(&q) && !known(w, q)? {
                    return Err(CatalogError::Invalid(format!("payload {id} names unknown parent {q}")));
                }
            }
            if let Some(mark) = p.coverage
                && !w.has(STREAMS, &k32(mark.source.0))?
            {
                return Err(CatalogError::NotFound(format!("coverage source stream {}", mark.source.0)));
            }
        }
        Ok(())
    }

    /// Writes a checked payload, returning its commit sequence.
    fn put(w: &mut W, p: NewPayload) -> Result<u64> {
        let h = p.header;
        let id = h.id();
        let stream = h.stream().0;
        let commit_seq = w.get(META, b"commit_seq")?.map_or(Ok(0), |b| codec::u64_at(&b, 0))? + 1;
        w.put(META, b"commit_seq", &commit_seq.to_be_bytes())?;
        w.put(PAYLOAD_SEQ, &id.0, &commit_seq.to_be_bytes())?;
        w.put(STREAM_SEQ, &Out::new().u32(stream).u64(commit_seq).done(), &id.0)?;
        let span = span_of(&h)?;
        let longest = w.get(STREAM_LONGEST, &k32(stream))?.map_or(Ok(0), |b| codec::u64_at(&b, 0))?;
        if span.len_ns() > longest {
            w.put(STREAM_LONGEST, &k32(stream), &span.len_ns().to_be_bytes())?;
        }
        w.put(TIME_INDEX, &k_time(stream, h.start_ns(), h.seq()), &id.0)?;
        let ps = h.parents();
        for &q in &ps {
            w.put(PROV_CHILDREN, &k_pair(q, id), &[])?;
        }
        if !ps.is_empty() {
            let flat: Vec<u8> = ps.iter().flat_map(|q| q.0).collect();
            w.put(PROV_PARENTS, &id.0, &flat)?;
        }
        let h_subjects = h.subjects();
        for &s in &h_subjects {
            w.put(SUBJECTS, &k_pair(s, id), &[])?;
        }
        if let Some(loc) = &p.location {
            put_location(w, id, loc)?;
        }
        if let Some(mark) = p.coverage {
            cov_insert(w, mark.source.0, mark.stage.0, mark.span)?;
        }
        w.put(PAYLOADS, &id.0, h.as_bytes())?;
        if recording_get(w, id)?.is_none()
            && let Some(info) = w.get(STREAMS, &k32(stream))?.map(|b| codec::read_stream_info(&b)).transpose()?
            && crate::recording::native(&info)
        {
            // An instant audio header is not a source-duration bound.
            if info.name.ends_with(".audio") && h.end_ns() <= h.start_ns() {
                let mut meta = recording_meta(w, h.stream())?;
                meta.current_known = false;
                recording_save_meta(w, h.stream(), &meta)?;
            }
            let row = crate::recording::Row::raw(&h, &info, recording_meta(w, h.stream())?.generation, commit_seq)?;
            recording_put(w, h.stream(), row)?;
        }
        // Derived from something a running purge hid, or copied from a chunk over a frame it cuts: hidden too.
        let mut by = None;
        for &q in &ps {
            if let Some(b) = w.get(DOOMED, &q.0)? {
                let purge = codec::id_at(&b, 0)?;
                // An age-out takes only what it names: a child of what it is removing stays.
                if load_job(w, purge)?.is_some_and(|j| j.selector.is_age_out()) {
                    continue;
                }
                by = Some(purge);
                break;
            }
            let mut hit = None;
            w.prefix(PURGE_CUTS, &q.0, |k, v| {
                if codec::interval(&mut In::new(v))?.overlaps(span) {
                    hit = Some(codec::id_at(k, 16)?);
                }
                Ok(hit.is_none())
            })?;
            if hit.is_some() {
                by = hit;
                break;
            }
        }
        if let Some(purge) = by
            && let Some(mut job) = load_job(w, purge)?
        {
            doom(w, purge, &mut job, id, true, &mut PurgeDelta::default())?;
            w.put(PURGE_JOBS, &purge.0, &job.encode())?;
        }
        // What a running purge's selector names is purged, wherever the purge's scan has got to: a payload or chunk
        // committed behind a range purge's cursor, or about an entity being purged, mustn't escape it.
        let mut jobs = Vec::new();
        w.scan(PURGE_JOBS, &[], None, &mut |k, v| {
            jobs.push((codec::id_at(k, 0)?, Job::decode(v)?));
            Ok(true)
        })?;
        for (purge, mut job) in jobs {
            let mut late = PurgeDelta::default();
            match job.selector.clone() {
                PurgeSelector::Range { span: range, .. } if job.streams.contains(&stream) && range.overlaps(span) => {
                    root_in_range(w, purge, &mut job, id, range, &mut late)?;
                }
                PurgeSelector::Entity(e) if h_subjects.contains(&e) => doom(w, purge, &mut job, id, false, &mut late)?,
                _ => continue,
            }
            keep_late(w, purge, &mut job, id, late)?;
            w.put(PURGE_JOBS, &purge.0, &job.encode())?;
        }
        Ok(commit_seq)
    }
}

/// What a chunk commit or part found about the frames it names, before anything is written.
struct Absorbing {
    /// The frames as they land: the caller's, with the ones a finished purge removed marked cut.
    frames: Vec<ChunkFrame>,
    /// The frames to absorb: each one's id, header, and index in the whole chunk.
    keep: Vec<(Uuid, HeaderBuf, u32)>,
    /// Frames whose rows stay (hidden by a purge, or cut by the worker) but whose slots the chunk lands without.
    cut_ids: Vec<Uuid>,
    /// The frames a running purge has hidden, with that purge: their slots are its cuts.
    doomed_cuts: Vec<(Uuid, u32)>,
}

/// Checks the frames `part` adds to chunk `id` from index `base` on (0 for a whole chunk), the last of the frames
/// before them starting at `previous`, against the payloads in `absorbed`.
#[allow(clippy::too_many_arguments)]
fn absorbing(
    w: &W,
    id: Uuid,
    stream: StreamId,
    chunk_span: Interval,
    base: usize,
    mut previous: u64,
    part: &[ChunkFrame],
    absorbed: &[Uuid],
) -> Result<Absorbing> {
    let mut frames = part.to_vec();
    let mut keep = Vec::new();
    // Frames whose rows stay (hidden by a purge, or cut by the worker) but whose slots the chunk lands without.
    let mut cut_ids: Vec<Uuid> = Vec::new();
    let mut doomed_cuts: Vec<(Uuid, u32)> = Vec::new();
    for (index, (frame, &f)) in part.iter().zip(absorbed).enumerate() {
        let index = base + index;
        if frame.start_ns < previous || !chunk_span.contains(frame.start_ns) {
            return Err(CatalogError::Invalid(format!(
                "chunk {id}'s frame {index} starts at {} (the chunk spans {chunk_span:?}), or goes backwards",
                frame.start_ns
            )));
        }
        previous = frame.start_ns;
        if f == id {
            return Err(CatalogError::Invalid(format!("chunk {id} absorbs itself")));
        }
        let Some(h) = header_of(w, f)? else {
            // Gone: cut when a purge removed it (its record says so); an id never indexed is a worker bug.
            if purged(w, f)? {
                frames[index - base].cut = true;
                continue;
            }
            return Err(CatalogError::Invalid(format!("chunk {id} absorbs {f}, not an indexed frame on its stream")));
        };
        if h.stream() != stream {
            return Err(CatalogError::Invalid(format!("chunk {id} absorbs {f}, not an indexed frame on its stream")));
        }
        if let Some(b) = w.get(DOOMED, &f.0)? {
            // Hidden by a running purge: the purge cuts the slot (below, once the chunk is in) and reports it,
            // as it does a chunk it finds in place; the row is the purge's to remove.
            cut_ids.push(f);
            doomed_cuts.push((codec::id_at(&b, 0)?, u32::try_from(index).unwrap_or(u32::MAX)));
            continue;
        }
        if w.has(PROV_PARENTS, &f.0)? {
            return Err(CatalogError::Invalid(format!("chunk {id} absorbs {f}, which has parents (not a raw frame)")));
        }
        // A payload with dependencies is derived as much as one with parents: absorbing it would
        // drop its header and commit sequence, and its frame row is pruned an hour on.
        let mut dependent = false;
        w.prefix(DEPS_OF, &f.0, |_, _| {
            dependent = true;
            Ok(false)
        })?;
        if dependent {
            return Err(CatalogError::Invalid(format!(
                "chunk {id} absorbs {f}, which has dependencies (not a raw frame)"
            )));
        }
        if frame.start_ns != h.start_ns() {
            return Err(CatalogError::Invalid(format!(
                "chunk {id}'s frame for {f} starts at {} (the frame starts at {})",
                frame.start_ns,
                h.start_ns()
            )));
        }
        if frame.cut {
            // The worker cut it (a purge reached the file before this commit): the frame stays its own
            // payload, and the chunk lands without its slot (see the memory catalog).
            cut_ids.push(f);
            continue;
        }
        keep.push((f, h, u32::try_from(index).unwrap_or(u32::MAX)));
    }
    Ok(Absorbing { frames, keep, cut_ids, doomed_cuts })
}

/// Removes the rows of the frames a chunk absorbs: the chunk's frame index and their `FRAMES` rows stand for them.
fn drop_absorbed(w: &mut W, keep: &[(Uuid, HeaderBuf, u32)]) -> Result<()> {
    for (f, h, _) in keep {
        w.del(PAYLOADS, &f.0)?;
        drop_seq(w, h.stream().0, *f)?;
        w.del(TIME_INDEX, &k_time(h.stream().0, h.start_ns(), h.seq()))?;
        for l in locations(w, *f)? {
            del_location(w, *f, l.tier())?;
        }
    }
    Ok(())
}

/// Every absorbed frame keeps a `FRAMES` row (its chunk, its index, its subjects).
fn put_frame_rows(w: &mut W, chunk: Uuid, keep: &[(Uuid, HeaderBuf, u32)]) -> Result<()> {
    for (f, h, index) in keep {
        let subjects = h.subjects();
        let value = Out::new().id(chunk).u32(*index).raw(&subjects.iter().flat_map(|s| s.0).collect::<Vec<_>>()).done();
        w.put(FRAMES, &f.0, &value)?;
        w.put(CHUNK_FRAMES, &k_pair(chunk, *f), &[])?;
    }
    Ok(())
}

/// What an insert reports about payload `id`: its sequence, and whether a running purge has it hidden.
fn inserted(rd: &impl Reader, id: Uuid, duplicate: bool) -> Result<Inserted> {
    inserted_as(rd, id, id, duplicate)
}

/// As `inserted`, with the sequence of `answers` (the chunk, for a repeat of a frame it absorbed).
fn inserted_as(rd: &impl Reader, id: Uuid, answers: Uuid, duplicate: bool) -> Result<Inserted> {
    let seq = rd.get(PAYLOAD_SEQ, &answers.0)?.map_or(Ok(0), |b| codec::u64_at(&b, 0))?;
    Ok(Inserted { seq, hidden: doomed(rd, id)? || doomed(rd, answers)?, duplicate, cut: Vec::new() })
}

/// The chunk that absorbed the frame `h` is a repeat of: `h`'s id has a `FRAMES` row whose chunk is on `h`'s stream
/// and holds a frame starting when `h` does. None for a header that isn't such a repeat (the caller checks it as new;
/// an absorbed frame's id with another start is a conflict there, as any reused id is).
fn absorbed_as(rd: &impl Reader, h: &HeaderBuf) -> Result<Option<Uuid>> {
    let Some((chunk, index, _)) = frame_row(rd, h.id())? else { return Ok(None) };
    let Some(owner) = header_of(rd, chunk)? else { return Ok(None) };
    let Some(frames) = chunk_frames(rd, chunk)? else { return Ok(None) };
    let same = owner.stream() == h.stream()
        && frames.get(index as usize).is_some_and(|f| f.start_ns == h.start_ns() && !f.cut);
    Ok(same.then_some(chunk))
}

/// Whether a payload's bytes are inline.
fn is_inline(rd: &impl Reader, id: Uuid) -> Result<bool> {
    Ok(locations(rd, id)?.iter().any(|l| matches!(l, StorageLocator::Inline { .. })))
}

/// Removes a payload's commit sequence rows (the counter itself never goes back).
fn drop_seq(w: &mut W, stream: u32, id: Uuid) -> Result<()> {
    if let Some(b) = w.del(PAYLOAD_SEQ, &id.0)? {
        w.del(STREAM_SEQ, &Out::new().u32(stream).u64(codec::u64_at(&b, 0)?).done())?;
    }
    Ok(())
}

// Four indexes of at most 8 MiB each. Oversized windows retain only a safe sequence lower bound.
const WINDOW_CACHE_ENTRIES: usize = 4;
#[cfg(not(test))]
const WINDOW_CACHE_ROWS: usize = 1_048_576;
#[cfg(test)]
const WINDOW_CACHE_ROWS: usize = 64;

#[derive(Default)]
struct WindowCache {
    entries: VecDeque<WindowSlot>,
}

struct WindowSlot {
    database: Weak<Database>,
    stream: StreamId,
    window: Interval,
    index: Arc<Mutex<Option<WindowIndex>>>,
}

impl WindowCache {
    fn slot(
        &mut self,
        database: &Arc<Database>,
        stream: StreamId,
        window: Interval,
    ) -> Arc<Mutex<Option<WindowIndex>>> {
        let identity = Arc::downgrade(database);
        let at =
            self.entries.iter().position(|i| i.database.ptr_eq(&identity) && i.stream == stream && i.window == window);
        let slot = at.and_then(|at| self.entries.remove(at)).unwrap_or_else(|| WindowSlot {
            database: identity,
            stream,
            window,
            index: Arc::new(Mutex::new(None)),
        });
        let index = Arc::clone(&slot.index);
        self.entries.push_front(slot);
        self.entries.truncate(WINDOW_CACHE_ENTRIES);
        index
    }
}

struct WindowIndex {
    stream: StreamId,
    window: Interval,
    head: u64,
    snapshot: u64,
    first: Option<u64>,
    // None means the window exceeded the memory budget: page the sequence index from `first` instead.
    sequences: Option<Vec<u64>>,
}

impl WindowIndex {
    fn add(&mut self, seq: u64) {
        self.first = Some(self.first.map_or(seq, |first| first.min(seq)));
        if let Some(rows) = &mut self.sequences {
            if rows.len() == WINDOW_CACHE_ROWS {
                self.sequences = None;
            } else {
                rows.push(seq);
            }
        }
    }

    fn build(r: &impl Reader, stream: StreamId, window: Interval, head: u64, snapshot: u64) -> Result<Self> {
        let mut index = Self { stream, window, head, snapshot, first: None, sequences: Some(Vec::new()) };
        let reach = r.get(STREAM_LONGEST, &k32(stream.0))?.map_or(Ok(0), |b| codec::u64_at(&b, 0))?;
        let lo = k_time(stream.0, window.start_ns().saturating_sub(reach), 0);
        let hi = k_time(stream.0, window.end_ns(), 0);
        r.scan(TIME_INDEX, &lo, Some(&hi), &mut |_, v| {
            let id = codec::id_at(v, 0)?;
            if header_of(r, id)?.is_some_and(|h| h.span().is_some_and(|s| s.overlaps(window)))
                && let Some(seq) = r.get(PAYLOAD_SEQ, &id.0)?.map(|b| codec::u64_at(&b, 0)).transpose()?
            {
                index.add(seq);
            }
            Ok(true)
        })?;
        if let Some(rows) = &mut index.sequences {
            rows.sort_unstable();
        }
        Ok(index)
    }

    fn extend(&mut self, r: &impl Reader, head: u64) -> Result<()> {
        if head <= self.head {
            return Ok(());
        }
        let lo = Out::new().u32(self.stream.0).u64(self.head + 1).done();
        let hi = prefix_end(&k32(self.stream.0));
        r.scan(STREAM_SEQ, &lo, hi.as_deref(), &mut |k, v| {
            let seq = codec::u64_at(k, 4)?;
            let id = codec::id_at(v, 0)?;
            if header_of(r, id)?.is_some_and(|h| h.span().is_some_and(|s| s.overlaps(self.window))) {
                self.add(seq); // newer commits append in sequence order, including late imported event times
            }
            Ok(true)
        })?;
        self.head = head;
        Ok(())
    }

    fn page(&self, r: &impl Reader, after: u64, filter: &QueryFilter, limit: usize) -> Result<SeqPage> {
        let mut items = Vec::new();
        let mut last = after;
        let mut more = false;
        let mut visit = |seq, id| -> Result<bool> {
            let Some(header) = header_of(r, id)? else { return Ok(true) };
            if !header.span().is_some_and(|s| s.overlaps(self.window)) || !keeps(&header, filter) {
                return Ok(true);
            }
            // A cut frame can remain reachable by id/sequence after its chunk takes the time-index key.
            let key = k_time(header.stream().0, header.start_ns(), header.seq());
            if r.get(TIME_INDEX, &key)?.as_deref() != Some(id.0.as_slice()) {
                return Ok(true);
            }
            if items.len() == limit {
                more = true;
                return Ok(false);
            }
            last = seq;
            // Visibility and copies are resolved in this page's snapshot. Reuse its validated header;
            // rereading the same PAYLOADS row would allocate and verify the identical bytes again.
            if !doomed(r, id)? {
                items.push(indexed_header(r, id, header)?);
            }
            Ok(true)
        };
        if let Some(rows) = &self.sequences {
            for &seq in &rows[rows.partition_point(|&seq| seq <= after)..] {
                let key = Out::new().u32(self.stream.0).u64(seq).done();
                if let Some(id) = r.get(STREAM_SEQ, &key)?.map(|b| codec::id_at(&b, 0)).transpose()?
                    && !visit(seq, id)?
                {
                    break;
                }
            }
        } else if let Some(first) = self.first {
            let lo = Out::new().u32(self.stream.0).u64((after + 1).max(first)).done();
            let hi = prefix_end(&k32(self.stream.0));
            r.scan(STREAM_SEQ, &lo, hi.as_deref(), &mut |k, v| visit(codec::u64_at(k, 4)?, codec::id_at(v, 0)?))?;
        }
        Ok(SeqPage { items, next_after: more.then_some(last) })
    }
}

/// Index a time window once, then seek its commit sequences across pages. The cache is only an acceleration:
/// current catalog rows determine visibility, and new commits extend it even when their timestamps are old.
fn windowed(
    r: &impl Reader,
    slot: &Mutex<Option<WindowIndex>>,
    stream: StreamId,
    snapshot: u64,
    after_seq: u64,
    filter: &QueryFilter,
    limit: usize,
) -> Result<SeqPage> {
    let window = filter.window.expect("windowed queries have a time window");
    let head = r.get(META, b"commit_seq")?.map_or(Ok(0), |b| codec::u64_at(&b, 0))?;
    let mut cached = slot.lock().unwrap_or_else(PoisonError::into_inner);
    if cached.as_ref().is_some_and(|i| i.snapshot > snapshot) {
        drop(cached);
        return WindowIndex::build(r, stream, window, head, snapshot)?.page(r, after_seq, filter, limit);
    }
    if cached.is_none() {
        *cached = Some(WindowIndex::build(r, stream, window, head, snapshot)?);
    }
    let index = cached.as_mut().expect("the window index was built");
    if let Err(error) = index.extend(r, head) {
        *cached = None; // a partial scan must not append the same sequences again on retry
        return Err(error);
    }
    index.snapshot = snapshot;
    index.page(r, after_seq, filter, limit)
}

fn keeps(h: &HeaderBuf, filter: &QueryFilter) -> bool {
    filter.ids.as_ref().is_none_or(|ids| ids.contains(&h.id()))
        && filter.start_before.is_none_or(|t| h.start_ns() < t)
        && filter.subjects.as_ref().is_none_or(|want| h.subjects().iter().any(|s| want.contains(s)))
}

fn same_identity(a: &StreamInfo, b: &StreamInfo) -> bool {
    (a.name.as_str(), a.kind, a.layer, a.format, a.shape_name.as_str(), a.shape_version)
        == (b.name.as_str(), b.kind, b.layer, b.format, b.shape_name.as_str(), b.shape_version)
}

impl Catalog for RedbCatalog {
    fn flush_durable(&self) -> Result<()> {
        self.flush()
    }

    fn restart_stamp(&self) -> Result<Option<crate::RestartStamp>> {
        self.read(|r| {
            let stamp = crate::restart::stamp(r)?
                .ok_or_else(|| CatalogError::Storage("the catalog is missing its restart tracking".into()))?;
            if !crate::restart::ready(r)? {
                return Err(CatalogError::Storage("restart hot index is not ready".into()));
            }
            Ok(Some(stamp))
        })
    }

    fn load_restart_projection(
        &self,
        max_bytes: usize,
    ) -> Result<std::result::Result<crate::RestartProjection, crate::RestartMiss>> {
        self.read(|r| crate::restart::load(r, max_bytes))
    }

    fn save_restart_projection(&self, expected: crate::RestartStamp, bytes: &[u8]) -> Result<bool> {
        self.write_restart(false, |w| crate::restart::publish(w, expected, bytes))
    }

    fn recording_page(
        &self,
        stream: StreamId,
        window: Interval,
        after: Option<crate::RecordingCursor>,
        expected_revision: Option<u64>,
        max_items: usize,
        max_scan: usize,
        max_bytes: usize,
    ) -> Result<crate::RecordingPage> {
        self.read(|r| {
            let info = r
                .get(STREAMS, &k32(stream.0))?
                .map(|b| codec::read_stream_info(&b))
                .transpose()?
                .ok_or_else(|| CatalogError::NotFound(format!("stream {}", stream.0)))?;
            if !crate::recording::native(&info) {
                return Err(crate::recording::invalid("source is not native audio"));
            }
            let revision = r.get(META, &invalidation_key(stream))?.map_or(Ok(0), |b| codec::u64_at(&b, 0))?;
            if expected_revision.is_some_and(|want| want != revision) {
                return Err(crate::recording::invalid("recording revision changed"));
            }
            let meta = recording_meta(r, stream)?;
            let through = r.get(META, b"commit_seq")?.map_or(Ok(0), |b| codec::u64_at(&b, 0))?;
            let mut page = crate::recording::PageBuilder::new(
                &meta,
                revision,
                through,
                recording_era(r)?,
                window,
                after,
                max_items,
                max_scan,
                max_bytes,
            )?;
            let start = after.map_or(window.start_ns().saturating_sub(meta.longest), |c| c.start_ns);
            let lo = after.map_or_else(
                || Out::new().u32(stream.0).u64(start).done(),
                |c| Out::new().u32(stream.0).u64(c.start_ns).id(c.id).done(),
            );
            let hi = Out::new().u32(stream.0).u64(window.end_ns()).done();
            r.scan(RECORDING_ROWS, &lo, Some(&hi), &mut |key, bytes| {
                let row = crate::recording::Row::decode(bytes)?;
                if recording_key(stream, &row) != key
                    || (!row.scope_only && r.get(RECORDING_IDS, &row.input.chunk_id.0)?.as_deref() != Some(key))
                {
                    return Err(crate::recording::invalid("recording indexes disagree"));
                }
                if !row.scope_only {
                    for subject in &row.subjects {
                        if !r.has(RECORDING_SUBJECTS, &k_pair(*subject, row.input.chunk_id))? {
                            return Err(crate::recording::invalid("missing original subject index"));
                        }
                    }
                }
                page.visit(row)
            })?;
            Ok(page.finish())
        })
    }

    fn register_stream(&self, info: StreamInfo) -> Result<StreamId> {
        self.write(|w| {
            if let Some(b) = w.get(STREAM_NAMES, info.name.as_bytes())? {
                let id = codec::u32_at(&b, 0)?;
                let known = w.get(STREAMS, &k32(id))?.map(|v| codec::read_stream_info(&v)).transpose()?;
                if !known.as_ref().is_some_and(|k| same_identity(k, &info)) {
                    return Err(CatalogError::Conflict(format!(
                        "stream {:?} is registered as something else",
                        info.name
                    )));
                }
                if crate::recording::native(&info) {
                    let mut meta = recording_meta(w, StreamId(id))?;
                    if known.as_ref().is_some_and(|k| k.anchor != info.anchor) {
                        meta.generation = meta
                            .generation
                            .checked_add(1)
                            .ok_or_else(|| crate::recording::invalid("source generation exhausted"))?;
                        invalidate(w, StreamId(id))?;
                        let generation = stream_generation(w)?
                            .checked_add(1)
                            .ok_or_else(|| crate::recording::invalid("registration generation exhausted"))?;
                        w.put(META, b"stream_generation", &generation.to_be_bytes())?;
                    }
                    recording_save_meta(w, StreamId(id), &meta)?;
                }
                w.put(STREAMS, &k32(id), &codec::stream_info(&info))?;
                return Ok(StreamId(id));
            }
            let mut last = 0;
            w.scan(STREAMS, &[], None, &mut |k, _| {
                last = codec::u32_at(k, 0)?;
                Ok(true)
            })?;
            let id = last + 1;
            if crate::recording::native(&info) {
                recording_save_meta(
                    w,
                    StreamId(id),
                    &crate::recording::Meta { known: true, current_known: true, generation: 1, longest: 0 },
                )?;
            }
            w.put(STREAM_NAMES, info.name.as_bytes(), &k32(id))?;
            w.put(STREAMS, &k32(id), &codec::stream_info(&info))?;
            let generation = stream_generation(w)?;
            w.put(
                META,
                b"stream_generation",
                &generation
                    .checked_add(1)
                    .ok_or_else(|| crate::recording::invalid("registration generation exhausted"))?
                    .to_be_bytes(),
            )?;
            Ok(StreamId(id))
        })
    }

    fn stream(&self, id: StreamId) -> Result<Option<StreamInfo>> {
        self.read(|r| r.get(STREAMS, &k32(id.0))?.map(|b| codec::read_stream_info(&b)).transpose())
    }

    fn invalidation_revision(&self, stream: StreamId) -> Result<Option<u64>> {
        self.read(|r| {
            if !r.has(STREAMS, &k32(stream.0))? {
                return Err(CatalogError::NotFound(format!("stream {}", stream.0)));
            }
            let revision = r.get(META, &invalidation_key(stream))?.map_or(Ok(0), |v| codec::u64_at(&v, 0))?;
            Ok((revision != u64::MAX).then_some(revision))
        })
    }

    fn forget_revision(&self, stream: StreamId) -> Result<Option<u64>> {
        self.read(|r| {
            if !r.has(STREAMS, &k32(stream.0))? {
                return Err(CatalogError::NotFound(format!("stream {}", stream.0)));
            }
            let now = revision_of(r, stream)?;
            if now == u64::MAX {
                return Ok(None);
            }
            r.get(META, &forgotten_key(stream))?.map_or(Ok(now), |v| codec::u64_at(&v, 0)).map(Some)
        })
    }

    fn stream_generation(&self) -> Result<Option<u64>> {
        self.read(|r| stream_generation(r).map(Some))
    }

    fn streams(&self, after: Option<StreamId>, limit: usize) -> Result<Vec<(StreamId, StreamInfo)>> {
        if limit == 0 {
            return Err(CatalogError::Invalid("a streams limit must be at least 1".into()));
        }
        self.read(|r| {
            let Some(from) = after.map_or(Some(0), |a| a.0.checked_add(1)) else { return Ok(Vec::new()) };
            let mut out = Vec::new();
            r.scan(STREAMS, &k32(from), None, &mut |k, v| {
                out.push((StreamId(codec::u32_at(k, 0)?), codec::read_stream_info(v)?));
                Ok(out.len() < limit)
            })?;
            Ok(out)
        })
    }

    fn coverage_keys(&self, after: Option<(StreamId, StageId)>, limit: usize) -> Result<Vec<(StreamId, StageId)>> {
        if limit == 0 {
            return Err(CatalogError::Invalid("a coverage keys limit must be at least 1".into()));
        }
        self.read(|r| {
            let lo = match after {
                Some((s, g)) => match prefix_end(&Out::new().u32(s.0).u32(g.0).done()) {
                    Some(lo) => lo,
                    None => return Ok(Vec::new()),
                },
                None => Vec::new(),
            };
            let mut out: Vec<(StreamId, StageId)> = Vec::new();
            r.scan(COVERAGE, &lo, None, &mut |k, _| {
                let pair = (StreamId(codec::u32_at(k, 0)?), StageId(codec::u32_at(k, 4)?));
                if out.last() != Some(&pair) {
                    out.push(pair);
                }
                Ok(out.len() <= limit)
            })?;
            out.truncate(limit);
            Ok(out)
        })
    }

    fn stream_id(&self, name: &str) -> Result<Option<StreamId>> {
        self.read(|r| r.get(STREAM_NAMES, name.as_bytes())?.map(|b| codec::u32_at(&b, 0).map(StreamId)).transpose())
    }

    fn insert(&self, payloads: Vec<NewPayload>) -> Result<Vec<Inserted>> {
        self.write(|w| Self::insert_rows(w, payloads, &std::collections::HashMap::new()))
    }

    fn insert_dependent(
        &self,
        payloads: Vec<NewPayload>,
        dependencies: Vec<(Uuid, Vec<Dependency>)>,
    ) -> Result<Vec<Inserted>> {
        let dependencies = dependencies
            .into_iter()
            .map(|(id, deps)| Ok((id, crate::types::normalize_dependencies(&deps)?)))
            .collect::<Result<Vec<_>>>()?;
        self.write(|w| {
            let ids: Vec<Uuid> = payloads.iter().map(|p| p.header.id()).collect();
            let given: std::collections::HashMap<Uuid, Vec<(u32, u64, u64)>> = dependencies
                .iter()
                .map(|(id, deps)| {
                    let mut set: Vec<_> =
                        deps.iter().map(|d| (d.stream.0, d.span.start_ns(), d.span.end_ns())).collect();
                    set.sort_unstable();
                    (*id, set)
                })
                .collect();
            let inserted = Self::insert_rows(w, payloads, &given)?;
            for (id, deps) in &dependencies {
                let Some(i) = ids.iter().position(|x| x == id) else {
                    return Err(CatalogError::Invalid(format!("a dependency of {id}, which isn't in this batch")));
                };
                // A repeat keeps the dependencies it was indexed with.
                if !inserted[i].duplicate {
                    put_dependencies(w, *id, deps)?;
                }
            }
            Ok(inserted)
        })
    }

    fn head_seq(&self) -> Result<u64> {
        self.read(|r| r.get(META, b"commit_seq")?.map_or(Ok(0), |b| codec::u64_at(&b, 0)))
    }

    fn query(
        &self,
        stream: StreamId,
        after_seq: u64,
        filter: &QueryFilter,
        limit: usize,
        max_scan: usize,
    ) -> Result<SeqPage> {
        if limit == 0 || max_scan == 0 {
            return Err(CatalogError::Invalid("a query's limit and scan cap must be at least 1".into()));
        }
        if let Some(window) = filter.window {
            // Order snapshot admission, not disk reads. A scrub cannot swap the database while this snapshot opens.
            let (database, r, snapshot) = {
                let guard = self.db.read().unwrap_or_else(PoisonError::into_inner);
                let db = guard.as_ref().ok_or_else(closed)?;
                let mut serial = self.window_snapshots.lock().unwrap_or_else(PoisonError::into_inner);
                let r = R::new(db.begin_read()?);
                *serial = serial
                    .checked_add(1)
                    .ok_or_else(|| CatalogError::Invalid("window snapshot counter exhausted".into()))?;
                (Arc::clone(db), r, *serial)
            };
            if !r.has(STREAMS, &k32(stream.0))? {
                return Err(CatalogError::NotFound(format!("stream {}", stream.0)));
            }
            if after_seq == u64::MAX {
                return Ok(SeqPage { items: Vec::new(), next_after: None });
            }
            let slot = self.windows.lock().unwrap_or_else(PoisonError::into_inner).slot(&database, stream, window);
            return windowed(&r, &slot, stream, snapshot, after_seq, filter, limit);
        }
        self.read(|r| {
            if !r.has(STREAMS, &k32(stream.0))? {
                return Err(CatalogError::NotFound(format!("stream {}", stream.0)));
            }
            let Some(from) = after_seq.checked_add(1) else {
                return Ok(SeqPage { items: Vec::new(), next_after: None });
            };
            let (mut items, mut scanned, mut last, mut more) = (Vec::new(), 0, after_seq, false);
            let hi = prefix_end(&k32(stream.0));
            r.scan(STREAM_SEQ, &Out::new().u32(stream.0).u64(from).done(), hi.as_deref(), &mut |k, v| {
                if items.len() == limit || scanned == max_scan {
                    more = true;
                    return Ok(false);
                }
                scanned += 1;
                last = codec::u64_at(k, 4)?;
                let id = codec::id_at(v, 0)?;
                if let Some(p) = indexed(r, id)?
                    && keeps(&p.header, filter)
                {
                    items.push(p);
                }
                Ok(true)
            })?;
            Ok(SeqPage { items, next_after: more.then_some(last) })
        })
    }

    fn get(&self, id: Uuid) -> Result<Option<Resolved>> {
        self.read(|r| {
            if let Some(p) = indexed(r, id)? {
                return Ok(Some(Resolved::Payload(p)));
            }
            if doomed(r, id)? {
                return Ok(None);
            }
            let Some((chunk, frame_index, _)) = frame_row(r, id)? else { return Ok(None) };
            Ok(indexed(r, chunk)?.map(|chunk| Resolved::AbsorbedFrame { chunk, frame_index }))
        })
    }

    fn range(&self, stream: StreamId, window: Interval, after: Option<RangeCursor>, limit: usize) -> Result<Page> {
        self.read(|r| range_page(r, stream, window, after, limit))
    }

    fn parents(&self, id: Uuid) -> Result<Vec<Uuid>> {
        self.read(|r| if doomed(r, id)? { Ok(Vec::new()) } else { parents(r, id) })
    }

    fn children(&self, id: Uuid) -> Result<Vec<Uuid>> {
        self.read(|r| {
            if doomed(r, id)? {
                return Ok(Vec::new());
            }
            let mut out = Vec::new();
            for c in children(r, id)? {
                if !doomed(r, c)? {
                    out.push(c);
                }
            }
            Ok(out)
        })
    }

    fn descendants(&self, roots: &[Uuid], limit: usize) -> Result<Vec<Uuid>> {
        if limit == 0 {
            return Err(CatalogError::Invalid("a descendants limit must be at least 1".into()));
        }
        self.read(|r| {
            let mut seen: BTreeSet<Uuid> = roots.iter().copied().collect();
            let mut queue: std::collections::VecDeque<Uuid> = std::collections::VecDeque::new();
            for &root in roots {
                if !doomed(r, root)? {
                    queue.push_back(root);
                }
            }
            let mut out = Vec::new();
            while let Some(id) = queue.pop_front() {
                let mut next = children(r, id)?;
                next.sort();
                for n in next {
                    if seen.insert(n) && !doomed(r, n)? {
                        out.push(n);
                        if out.len() == limit {
                            return Ok(out);
                        }
                        queue.push_back(n);
                    }
                }
            }
            Ok(out)
        })
    }

    fn set_location(&self, id: Uuid, locator: StorageLocator) -> Result<()> {
        self.write(|w| {
            if !w.has(PAYLOADS, &id.0)? {
                return Err(CatalogError::NotFound(format!("payload {id}")));
            }
            if doomed(w, id)? {
                return Err(CatalogError::Conflict(format!("payload {id} is being purged: no new copies")));
            }
            if is_inline(w, id)? || matches!(locator, StorageLocator::Inline { .. }) {
                return Err(CatalogError::Invalid(format!(
                    "payload {id}'s inline bytes are set at insert and never move"
                )));
            }
            put_location(w, id, &locator)
        })
    }

    fn clear_location(&self, id: Uuid, tier: StorageTier) -> Result<()> {
        self.write(|w| {
            if !w.has(PAYLOADS, &id.0)? {
                return Err(CatalogError::NotFound(format!("payload {id}")));
            }
            if doomed(w, id)? {
                return Err(CatalogError::Conflict(format!("payload {id} is being purged")));
            }
            if is_inline(w, id)? && tier == StorageTier::Tier1Warm {
                return Err(CatalogError::Invalid(format!("payload {id}'s inline bytes are never cleared")));
            }
            del_location(w, id, tier)?;
            if locations(w, id)?.is_empty() {
                recording_unknown(w, id)?;
            }
            Ok(())
        })
    }

    fn clear_locations(&self, ids: &[Uuid], tier: StorageTier) -> Result<usize> {
        self.write(|w| {
            let mut cleared = 0;
            for &id in ids {
                if !w.has(PAYLOADS, &id.0)? || doomed(w, id)? || (is_inline(w, id)? && tier == StorageTier::Tier1Warm) {
                    continue;
                }
                del_location(w, id, tier)?;
                if locations(w, id)?.is_empty() {
                    recording_unknown(w, id)?;
                }
                cleared += 1;
            }
            Ok(cleared)
        })
    }

    fn commit_chunk(&self, chunk: NewChunk, absorbed: &[Uuid]) -> Result<Inserted> {
        self.commit_chunk_with_provenance(chunk, absorbed, None)
    }

    fn commit_chunk_with_provenance(
        &self,
        chunk: NewChunk,
        absorbed: &[Uuid],
        provenance: Option<&crate::ChunkProvenance>,
    ) -> Result<Inserted> {
        let id = chunk.payload.header.id();
        let stream = chunk.payload.header.stream();
        if let Some(p) = provenance {
            p.validate(chunk.frames.len())?;
        }
        if chunk.frames.len() != absorbed.len() {
            return Err(CatalogError::Invalid(format!(
                "chunk {id} indexes {} frames but absorbs {}",
                chunk.frames.len(),
                absorbed.len()
            )));
        }
        if absorbed.len() > MAX_CHUNK_FRAMES {
            return Err(CatalogError::Invalid(format!("chunk {id} absorbs more than {MAX_CHUNK_FRAMES} frames")));
        }
        match &chunk.payload.location {
            None => return Err(CatalogError::Invalid(format!("chunk {id} has no location"))),
            Some(l) if l.tier() == StorageTier::Tier0Hot || matches!(l, StorageLocator::Inline { .. }) => {
                return Err(CatalogError::Invalid(format!(
                    "chunk {id} must be placed in a chunk file, in Tier 1 or 2"
                )));
            }
            Some(_) => {}
        }
        if absorbed.iter().collect::<BTreeSet<_>>().len() != absorbed.len() {
            return Err(CatalogError::Invalid(format!("chunk {id} absorbs a frame twice")));
        }
        let chunk_span = span_of(&chunk.payload.header)?;
        self.write(|w| {
            // Committed already with identical header bytes (a worker retrying after a lost reply): idempotent, like a
            // repeated insert. The frames it absorbed are compacted by now, so the checks below would refuse it.
            if w.get(PAYLOADS, &id.0)?.as_deref() == Some(chunk.payload.header.as_bytes()) {
                return inserted(w, id, true);
            }
            // A frame a purge is removing (doomed) or has removed (no row left) becomes a cut frame of the chunk
            // rather than refusing it (see the memory catalog's commit_chunk for the reasoning).
            let recording_info = w
                .get(STREAMS, &k32(stream.0))?
                .map(|b| codec::read_stream_info(&b))
                .transpose()?
                .filter(crate::recording::native);
            let original_recordings = absorbed
                .iter()
                .map(|id| {
                    if let Some(row) = recording_get(w, *id)? {
                        return Ok(Some(row));
                    }
                    let Some(h) = header_of(w, *id)? else { return Ok(None) };
                    let seq = w.get(PAYLOAD_SEQ, &id.0)?.map_or(Ok(0), |b| codec::u64_at(&b, 0))?;
                    if seq >= recording_era(w)? {
                        return Ok(None);
                    }
                    recording_info.as_ref().map(|info| crate::recording::Row::raw(&h, info, 0, seq)).transpose()
                })
                .collect::<Result<Vec<_>>>()?;
            let Absorbing { frames, keep, cut_ids, doomed_cuts } =
                absorbing(w, id, stream, chunk_span, 0, 0, &chunk.frames, absorbed)?;
            // The frames' (stream, start, seq) keys are freed by this commit, so the chunk may take one.
            drop_absorbed(w, &keep)?;
            // A cut frame keeps its row, and the chunk's own key (its first frame's start, seq 0) may be that frame's:
            // the chunk takes it, and the frame stays reachable by id and sequence.
            let own = &chunk.payload.header;
            let key = k_time(stream.0, own.start_ns(), own.seq());
            if let Some(holder) = w.get(TIME_INDEX, &key)?
                && cut_ids.contains(&codec::id_at(&holder, 0)?)
            {
                w.del(TIME_INDEX, &key)?;
            }
            Self::check(w, std::slice::from_ref(&chunk.payload))?;
            // Every absorbed frame keeps its row: a consumer slower than the tier worker (ASR, a VLM) commits a child a
            // second or more after its parent frame was absorbed, and the parent must still resolve.
            put_frame_rows(w, id, &keep)?;
            // What landed cut with no purge to report it (gone frames, the worker's cuts); a running purge's cuts,
            // made below, come in its next delta.
            let unreported = cut_indexes(&frames);
            w.put(CHUNKS, &id.0, &codec::frames(&frames))?;
            let recording_header = chunk.payload.header.clone();
            Self::put(w, chunk.payload)?;
            if let Some(info) = recording_info {
                let clean = !frames.iter().any(|f| f.cut)
                    && !doomed(w, id)?
                    && !recording_get(w, id)?.is_some_and(|row| row.unknown);
                let row = crate::recording::bind(
                    &recording_header,
                    &info,
                    recording_meta(w, stream)?.generation,
                    w.get(PAYLOAD_SEQ, &id.0)?.map_or(Ok(0), |b| codec::u64_at(&b, 0))?,
                    &original_recordings,
                    clean,
                )?;
                let era = recording_era(w)?;
                let mixed = row.input.first_seq < era;
                for &raw in absorbed {
                    if recording_get(w, raw)?
                        .is_some_and(|r| !r.bound && !r.unknown && (!mixed || r.input.first_seq < era))
                    {
                        recording_remove(w, raw)?;
                    }
                }
                recording_put(w, stream, row)?;
            }
            // The slots of the frames a purge had hidden are its cuts (a range purge's window reached them in `put`
            // already; an entity purge's didn't).
            if !doomed(w, id)? {
                for (purge, index) in doomed_cuts {
                    let Some(mut job) = load_job(w, purge)? else { continue };
                    let mut late = PurgeDelta::default();
                    cut(w, purge, &mut job, id, index, &mut late)?;
                    keep_late(w, purge, &mut job, id, late)?;
                    w.put(PURGE_JOBS, &purge.0, &job.encode())?;
                }
            }
            if let Some(p) = provenance {
                w.put(CHUNK_PROVENANCE, &id.0, &p.encode())?;
            }
            let mut done = inserted(w, id, false)?;
            done.cut = unreported;
            Ok(done)
        })
    }

    fn chunk_provenance(&self, chunk: Uuid) -> Result<Option<crate::ChunkProvenance>> {
        self.read(|r| {
            if doomed(r, chunk)? {
                return Ok(None);
            }
            r.get(CHUNK_PROVENANCE, &chunk.0)?.map(|b| crate::ChunkProvenance::decode(&b)).transpose()
        })
    }

    fn extend_chunk(
        &self,
        part: NewChunk,
        first_frame: u32,
        absorbed: &[Uuid],
        provenance: Option<&crate::ChunkProvenance>,
    ) -> Result<Inserted> {
        let header = part.payload.header.clone();
        let (id, stream, base) = (header.id(), header.stream(), first_frame as usize);
        crate::check_part(&part, base, absorbed, provenance)?;
        let span = span_of(&header)?;
        self.write(|w| {
            let (Some(known), Some(mut frames)) = (header_of(w, id)?, chunk_frames(w, id)?) else {
                return Err(CatalogError::NotFound(format!("chunk {id}")));
            };
            // The same part again (a worker retrying after a lost reply): its frames are compacted by now.
            if known.as_bytes() == header.as_bytes() && frames.len() == base + part.frames.len() {
                return inserted(w, id, true);
            }
            if doomed(w, id)? {
                return Err(CatalogError::Conflict(format!("chunk {id} is being purged")));
            }
            crate::check_grows(&known, &header, frames.len(), base)?;
            let previous = frames.last().map_or(0, |f| f.start_ns);
            let Absorbing { frames: tail, keep, cut_ids: _, doomed_cuts } =
                absorbing(w, id, stream, span, base, previous, &part.frames, absorbed)?;
            let recording_info = w
                .get(STREAMS, &k32(stream.0))?
                .map(|b| codec::read_stream_info(&b))
                .transpose()?
                .filter(crate::recording::native);
            let original_recordings = absorbed.iter().map(|raw| recording_get(w, *raw)).collect::<Result<Vec<_>>>()?;
            drop_absorbed(w, &keep)?;
            put_frame_rows(w, id, &keep)?;
            let unreported: Vec<u32> = cut_indexes(&tail).into_iter().map(|i| i + first_frame).collect();
            frames.extend(tail);
            w.put(CHUNKS, &id.0, &codec::frames(&frames))?;
            // The chunk's own rows: its header with the new end, its longer file, and a new place in commit order,
            // so a reader following the stream by commit sequence sees it again.
            w.put(PAYLOADS, &id.0, header.as_bytes())?;
            let longest = w.get(STREAM_LONGEST, &k32(stream.0))?.map_or(Ok(0), |b| codec::u64_at(&b, 0))?;
            if span.len_ns() > longest {
                w.put(STREAM_LONGEST, &k32(stream.0), &span.len_ns().to_be_bytes())?;
            }
            if let Some(location) = &part.payload.location {
                put_location(w, id, location)?;
            }
            drop_seq(w, stream.0, id)?;
            let commit_seq = w.get(META, b"commit_seq")?.map_or(Ok(0), |b| codec::u64_at(&b, 0))? + 1;
            w.put(META, b"commit_seq", &commit_seq.to_be_bytes())?;
            w.put(PAYLOAD_SEQ, &id.0, &commit_seq.to_be_bytes())?;
            w.put(STREAM_SEQ, &Out::new().u32(stream.0).u64(commit_seq).done(), &id.0)?;
            let stored = w.get(CHUNK_PROVENANCE, &id.0)?.map(|b| crate::ChunkProvenance::decode(&b)).transpose()?;
            if let Some(whole) = crate::ChunkProvenance::joined(stored, base, provenance, part.frames.len()) {
                w.put(CHUNK_PROVENANCE, &id.0, &whole.encode())?;
            }
            if let Some(info) = recording_info {
                let clean = !frames.iter().any(|f| f.cut);
                let row = crate::recording::extend(
                    recording_get(w, id)?,
                    &header,
                    &info,
                    recording_meta(w, stream)?.generation,
                    commit_seq,
                    &original_recordings,
                    clean,
                )?;
                let era = recording_era(w)?;
                let mixed = row.input.first_seq < era;
                for &raw in absorbed {
                    if recording_get(w, raw)?
                        .is_some_and(|r| !r.bound && !r.unknown && (!mixed || r.input.first_seq < era))
                    {
                        recording_remove(w, raw)?;
                    }
                }
                recording_put(w, stream, row)?;
            }
            for (purge, index) in doomed_cuts {
                let Some(mut job) = load_job(w, purge)? else { continue };
                let mut late = PurgeDelta::default();
                cut(w, purge, &mut job, id, index, &mut late)?;
                keep_late(w, purge, &mut job, id, late)?;
                w.put(PURGE_JOBS, &purge.0, &job.encode())?;
            }
            let mut done = inserted(w, id, false)?;
            done.cut = unreported;
            Ok(done)
        })
    }

    fn rewrite_chunk(&self, chunk: Uuid, frames: Vec<ChunkFrame>, location: StorageLocator) -> Result<()> {
        if location.tier() == StorageTier::Tier0Hot || matches!(location, StorageLocator::Inline { .. }) {
            return Err(CatalogError::Invalid(format!("chunk {chunk} must be placed in a chunk file, in Tier 1 or 2")));
        }
        self.write(|w| {
            if doomed(w, chunk)? {
                return Err(CatalogError::Conflict(format!("chunk {chunk} is being purged")));
            }
            let Some(known) = chunk_frames(w, chunk)? else {
                return Err(CatalogError::NotFound(format!("chunk {chunk}")));
            };
            let same = known.len() == frames.len()
                && known.iter().zip(&frames).all(|(a, b)| a.start_ns == b.start_ns && a.cut == b.cut);
            if !same {
                return Err(CatalogError::Invalid(format!("chunk {chunk} rewritten with different frames")));
            }
            w.put(CHUNKS, &chunk.0, &codec::frames(&frames))?;
            put_location(w, chunk, &location)
        })
    }

    fn chunk_snapshot(&self, chunk: Uuid) -> Result<Option<ChunkSnapshot>> {
        self.read(|r| {
            let Some(indexed) = indexed(r, chunk)? else {
                return Ok(None);
            };
            if indexed.frames.is_none() {
                return Err(CatalogError::Invalid("the indexed payload is not a chunk".into()));
            }
            let provenance =
                r.get(CHUNK_PROVENANCE, &chunk.0)?.map(|b| crate::ChunkProvenance::decode(&b)).transpose()?;
            Ok(Some(ChunkSnapshot { indexed, provenance }))
        })
    }

    fn storage_stats(&self) -> Result<Option<StorageStats>> {
        use redb::{ReadableTableMetadata, TableHandle};
        use std::os::unix::fs::MetadataExt;
        let mut tables = self.read(|r| {
            let mut out = Vec::new();
            for handle in r.txn.list_tables()? {
                let t = r.txn.open_untyped_table(handle.clone())?;
                let s = t.stats()?;
                out.push(TableStorage {
                    name: handle.name().to_owned(),
                    rows: t.len()?,
                    height: s.tree_height(),
                    leaf_pages: s.leaf_pages(),
                    branch_pages: s.branch_pages(),
                    stored_bytes: s.stored_bytes(),
                    metadata_bytes: s.metadata_bytes(),
                    fragmented_bytes: s.fragmented_bytes(),
                });
            }
            Ok(out)
        })?;
        tables.sort_by_key(|t| std::cmp::Reverse(t.leaf_pages + t.branch_pages));
        let meta = std::fs::metadata(&self.path).map_err(|e| CatalogError::Storage(Box::new(e)))?;
        // redb's default page size, which this catalog never changes.
        Ok(Some(StorageStats { page_size: 4096, file_bytes: meta.len(), disk_bytes: meta.blocks() * 512, tables }))
    }

    fn record_aged_out(&self, stream: StreamId, spans: &[Interval]) -> Result<()> {
        self.write(|w| {
            if !w.has(STREAMS, &k32(stream.0))? {
                return Err(CatalogError::NotFound(format!("stream {}", stream.0)));
            }
            for s in spans {
                aged_insert(w, stream.0, *s, 0)?;
            }
            Ok(())
        })
    }

    fn aged_out(&self, stream: StreamId, window: Interval, max: usize) -> Result<Vec<Interval>> {
        if max == 0 {
            return Err(CatalogError::Invalid("an aged-out listing's max must be at least 1".into()));
        }
        self.read(|r| aged_within(r, stream.0, window, max))
    }

    fn mark_commit_clock(&self, now_ns: u64) -> Result<()> {
        self.write(|w| {
            if let Some((k, _)) = w.last_below(COMMIT_CLOCK, &[], &u64::MAX.to_be_bytes())?
                && codec::u64_at(&k, 0)? >= now_ns
            {
                return Ok(());
            }
            let head = w.get(META, b"commit_seq")?.map_or(Ok(0), |b| codec::u64_at(&b, 0))?;
            w.put(COMMIT_CLOCK, &now_ns.to_be_bytes(), &head.to_be_bytes())
        })
    }

    fn commit_seq_at(&self, at_ns: u64) -> Result<u64> {
        self.read(|r| {
            let Some(below) = at_ns.checked_add(1) else {
                return r
                    .last_below(COMMIT_CLOCK, &[], &u64::MAX.to_be_bytes())?
                    .map_or(Ok(0), |(_, v)| codec::u64_at(&v, 0));
            };
            r.last_below(COMMIT_CLOCK, &[], &below.to_be_bytes())?.map_or(Ok(0), |(_, v)| codec::u64_at(&v, 0))
        })
    }

    fn add_coverage(&self, mark: CoverageMark) -> Result<()> {
        self.write(|w| {
            if !w.has(STREAMS, &k32(mark.source.0))? {
                return Err(CatalogError::NotFound(format!("stream {}", mark.source.0)));
            }
            cov_insert(w, mark.source.0, mark.stage.0, mark.span)
        })
    }

    fn coverage(&self, source: StreamId, stage: StageId, window: Interval) -> Result<Vec<Interval>> {
        self.read(|r| cov_within(r, source.0, stage.0, window))
    }

    fn coverage_gaps(&self, source: StreamId, stage: StageId, window: Interval, max: usize) -> Result<Vec<Interval>> {
        self.read(|r| cov_gaps(r, source.0, stage.0, window, max))
    }

    fn start_purge(
        &self,
        selector: PurgeSelector,
        reason: &str,
        detail: &str,
        purge_id: Uuid,
        now_ns: u64,
    ) -> Result<()> {
        self.write(|w| {
            if w.has(PURGES, &purge_id.0)? {
                return Err(CatalogError::Conflict(format!("purge {purge_id} is recorded already")));
            }
            let streams = match &selector {
                PurgeSelector::Range { streams: Some(list), .. } => {
                    for s in list {
                        if !w.has(STREAMS, &k32(s.0))? {
                            return Err(CatalogError::NotFound(format!("stream {}", s.0)));
                        }
                    }
                    list.iter().map(|s| s.0).collect()
                }
                PurgeSelector::Range { streams: None, .. } => {
                    let mut all = Vec::new();
                    w.scan(STREAMS, &[], None, &mut |k, _| {
                        all.push(codec::u32_at(k, 0)?);
                        Ok(true)
                    })?;
                    all
                }
                PurgeSelector::Entity(_) | PurgeSelector::Ids(_) | PurgeSelector::AgeOut { .. } => Vec::new(),
            };
            if let PurgeSelector::Range { span, .. } = &selector {
                let cutoff = w.get(META, b"commit_seq")?.map_or(Ok(0), |b| codec::u64_at(&b, 0))?;
                w.put(RECORDING_PURGE, &purge_id.0, &Out::new().u8(0).u64(cutoff).u32(0).bytes(&[]).done())?;
                for &stream in &streams {
                    recording_scope(w, StreamId(stream), *span, purge_id, recording_era(w)?)?;
                }
            }
            let kind = match &selector {
                PurgeSelector::Range { span, streams } => PurgeKind::Range { span: *span, streams: streams.clone() },
                PurgeSelector::Entity(_) => PurgeKind::Entity,
                PurgeSelector::Ids(ids) => PurgeKind::Ids { count: ids.len() },
                PurgeSelector::AgeOut { ids, .. } => PurgeKind::AgeOut { count: ids.len() },
            };
            let record = PurgeRecord {
                at_ns: now_ns,
                kind,
                reason: reason.to_owned(),
                detail: detail.to_owned(),
                report: None,
                completed_ns: None,
                acked: Vec::new(),
            };
            w.put(PURGES, &purge_id.0, &codec::record(&record))?;
            let seq = w.get(META, b"purge_seq")?.map_or(Ok(0), |b| codec::u64_at(&b, 0))?;
            w.put(META, b"purge_seq", &(seq + 1).to_be_bytes())?;
            w.put(PURGE_ORDER, &seq.to_be_bytes(), &purge_id.0)?;
            let selector = match selector {
                PurgeSelector::Ids(ids) => {
                    for (i, id) in (0u64..).zip(&ids) {
                        w.put(PURGE_IDS, &k_seq(purge_id, i), &id.0)?;
                    }
                    PurgeSelector::Ids(Vec::new())
                }
                PurgeSelector::AgeOut { ids, committed_by_seq, merge_gap_ns } => {
                    for (i, id) in (0u64..).zip(&ids) {
                        w.put(PURGE_IDS, &k_seq(purge_id, i), &id.0)?;
                    }
                    PurgeSelector::AgeOut { ids: Vec::new(), committed_by_seq, merge_gap_ns }
                }
                other => other,
            };
            let job = Job { selector, phase: Phase::Roots, streams, pos: 0, after: None, next_seq: 0, done: 0 };
            w.put(PURGE_JOBS, &purge_id.0, &job.encode())
        })
    }

    fn purge_step(&self, purge_id: Uuid, budget: usize) -> Result<PurgeProgress> {
        self.purge_work(purge_id, budget, None)
    }

    fn purge_batch(
        &self,
        purge_id: Uuid,
        max_rows: usize,
        should_yield: &mut dyn FnMut() -> bool,
    ) -> Result<PurgeProgress> {
        self.purge_work(purge_id, max_rows, Some(should_yield))
    }

    fn purge_record(&self, purge_id: Uuid) -> Result<Option<PurgeRecord>> {
        self.read(|r| record_of(r, purge_id))
    }

    fn purges(&self, after: Option<Uuid>, limit: usize) -> Result<Vec<PurgeSummary>> {
        if limit == 0 {
            return Err(CatalogError::Invalid("a purges limit must be at least 1".into()));
        }
        self.read(|r| {
            let mut ids = Vec::new();
            let mut seen = after.is_none();
            r.scan(PURGE_ORDER, &[], None, &mut |_, v| {
                let id = codec::id_at(v, 0)?;
                if seen {
                    ids.push(id);
                } else {
                    seen = Some(id) == after;
                }
                Ok(ids.len() < limit)
            })?;
            if !seen {
                return Err(CatalogError::NotFound(format!(
                    "purge {}",
                    after.map_or_else(String::new, |a| a.to_string())
                )));
            }
            let mut out = Vec::new();
            for id in ids {
                // The record without its report: a listing never reads the purged rows.
                let Some(b) = r.get(PURGES, &id.0)? else { continue };
                let rec = codec::read_record(&b)?;
                out.push(PurgeSummary {
                    purge_id: id,
                    at_ns: rec.at_ns,
                    kind: rec.kind,
                    reason: rec.reason,
                    detail: rec.detail,
                    removed: r.get(PURGE_COUNTS, &id.0)?.map_or(Ok(0), |b| codec::u64_at(&b, 0))?,
                    finished: rec.report.is_some(),
                    completed_ns: rec.completed_ns,
                });
            }
            Ok(out)
        })
    }

    fn pending_purges(&self) -> Result<Vec<Uuid>> {
        self.read(|r| {
            let mut ids = Vec::new();
            r.scan(PURGE_ORDER, &[], None, &mut |_, v| {
                ids.push(codec::id_at(v, 0)?);
                Ok(true)
            })?;
            let mut out = Vec::new();
            for id in ids {
                if let Some(b) = r.get(PURGES, &id.0)? {
                    let rec = codec::read_record(&b)?;
                    if rec.report.is_none() || rec.completed_ns.is_none() {
                        out.push(id);
                    }
                }
            }
            Ok(out)
        })
    }

    fn complete_purge(&self, purge_id: Uuid, now_ns: u64) -> Result<()> {
        self.write(|w| {
            let b = w.get(PURGES, &purge_id.0)?.ok_or_else(|| CatalogError::NotFound(format!("purge {purge_id}")))?;
            let mut rec = codec::read_record(&b)?;
            if rec.report.is_none() {
                return Err(CatalogError::Invalid(format!("purge {purge_id} is still running")));
            }
            if rec.completed_ns.is_some() {
                return Ok(());
            }
            rec.completed_ns.get_or_insert(now_ns);
            if matches!(rec.kind, PurgeKind::AgeOut { .. }) {
                // Every keeper has its notice: an age-out's per-item report collapses to its spans, which AGED_OUT
                // already holds, and its count, which the record keeps. A forget or delete keeps its exact report.
                let mut rows = Vec::new();
                w.scan(PURGE_REMOVED, &purge_id.0, prefix_end(&purge_id.0).as_deref(), &mut |k, _| {
                    rows.push(k.to_vec());
                    Ok(true)
                })?;
                for k in rows {
                    w.del(PURGE_REMOVED, &k)?;
                }
            }
            w.put(PURGES, &purge_id.0, &codec::record(&rec))
        })
    }

    fn keepers(&self) -> Result<Vec<String>> {
        self.read(keepers_of)
    }

    fn add_keeper(&self, name: &str) -> Result<()> {
        self.write(|w| {
            let mut names = keepers_of(w)?;
            if names.iter().any(|n| n == name) {
                return Ok(());
            }
            names.push(name.to_owned());
            names.sort();
            put_keepers(w, &names)
        })
    }

    fn forget_keeper(&self, name: &str) -> Result<bool> {
        self.write(|w| {
            let mut names = keepers_of(w)?;
            let before = names.len();
            names.retain(|n| n != name);
            if names.len() == before {
                return Ok(false);
            }
            put_keepers(w, &names)?;
            Ok(true)
        })
    }

    fn purge_acked(&self, purge_id: Uuid, keeper: &str) -> Result<()> {
        self.write(|w| {
            let b = w.get(PURGES, &purge_id.0)?.ok_or_else(|| CatalogError::NotFound(format!("purge {purge_id}")))?;
            let mut rec = codec::read_record(&b)?;
            if rec.acked.iter().any(|n| n == keeper) {
                return Ok(());
            }
            rec.acked.push(keeper.to_owned());
            w.put(PURGES, &purge_id.0, &codec::record(&rec))
        })
    }

    fn request_scrub(&self) -> Result<()> {
        self.write(|w| {
            bump_purge_gen(w)?;
            w.put(META, b"scrub_due", &[1])
        })
    }

    fn scrub_needed(&self) -> Result<bool> {
        self.read(|r| Ok(r.get(META, b"scrub_due")?.is_some_and(|v| v == [1])))
    }

    fn prune_frames(&self, before_ns: u64, after: Option<(Uuid, Uuid)>, budget: usize) -> Result<PruneProgress> {
        self.prune_frames_until(before_ns, after, budget, &mut || false)
    }

    fn prune_frames_until(
        &self,
        before_ns: u64,
        after: Option<(Uuid, Uuid)>,
        budget: usize,
        should_stop: &mut dyn FnMut() -> bool,
    ) -> Result<PruneProgress> {
        if budget == 0 {
            return Err(CatalogError::Invalid("a prune step's budget must be at least 1".into()));
        }
        self.write(|w| {
            // The next `budget` absorbed frames after the cursor, by (chunk, frame).
            let lo = after.map(|(c, f)| k_pair(c, f)).unwrap_or_default();
            let mut keys: Vec<(Uuid, Uuid)> = Vec::new();
            w.scan(CHUNK_FRAMES, &lo, None, &mut |k, _| {
                if after.is_some() && k == lo.as_slice() {
                    return Ok(true);
                }
                let mut i = In::new(k);
                keys.push((i.id()?, i.id()?));
                Ok(keys.len() < budget)
            })?;
            let mut pruned = 0;
            let mut looked = 0;
            // Each chunk's end, read once: its frames are past the window when the chunk is.
            let mut chunk_end: Option<(Uuid, Option<u64>)> = None;
            for &(chunk, frame) in &keys {
                if looked > 0 && should_stop() {
                    break;
                }
                looked += 1;
                let end = match chunk_end {
                    Some((c, end)) if c == chunk => end,
                    _ => {
                        let end = header_of(w, chunk)?.map(|h| h.end_ns());
                        chunk_end = Some((chunk, end));
                        end
                    }
                };
                // Young, or a chunk being purged (its rows are the purge's to remove).
                if end.is_none_or(|end| end >= before_ns) || doomed(w, chunk)? || doomed(w, frame)? {
                    continue;
                }
                let Some((_, _, subjects)) = frame_row(w, frame)? else { continue };
                let mut has_children = false;
                w.prefix(PROV_CHILDREN, &frame.0, |_, _| {
                    has_children = true;
                    Ok(false)
                })?;
                if has_children || !subjects.is_empty() {
                    continue; // lineage and purges by entity still need it
                }
                w.del(FRAMES, &frame.0)?;
                w.del(CHUNK_FRAMES, &k_pair(chunk, frame))?;
                pruned += 1;
            }
            // Stopped early: carry on after the last frame looked at. Otherwise, a short page is the end.
            let next = if looked < keys.len() {
                Some(keys[looked - 1])
            } else if keys.len() < budget {
                None
            } else {
                keys.last().copied()
            };
            Ok(PruneProgress { scanned: looked, pruned, next })
        })
    }

    fn scrub_step(&self, budget: usize) -> Result<ScrubProgress> {
        if budget == 0 {
            return Err(CatalogError::Invalid("a scrub step's budget must be at least 1".into()));
        }
        let mut guard = self.scrub.lock().unwrap_or_else(PoisonError::into_inner);
        if guard.is_none() {
            if !self.scrub_needed()? {
                return Ok(ScrubProgress::Idle);
            }
            // A marker left by the last swap (its removal failed or was lost) must be gone before a new fresh file
            // exists, or a crash now would put a half-copied file in place.
            let done = Self::done_path(&self.path);
            if done.exists() {
                std::fs::remove_file(&done).map_err(io_error)?;
                sync_dir(&self.path)?;
            }
            let tmp = Self::scrub_path(&self.path);
            let _ = std::fs::remove_file(&tmp);
            let db = Database::create(&tmp)?;
            let mut txn = db.begin_write()?;
            txn.set_durability(Durability::None)?;
            for &def in ALL {
                txn.open_table(def)?;
            }
            txn.commit()?;
            self.touched.lock().unwrap_or_else(PoisonError::into_inner).clear();
            self.scrubbing.store(true, Ordering::Release);
            let started_gen = self.read(purge_gen)?;
            *guard = Some(Scrub { tmp, db, started_gen, table: 0, after: None, copied: 0, bytes: 0, flushed: false });
            crash_point("started");
            return Ok(ScrubProgress::Copying { copied: 0 });
        }
        let Some(s) = guard.as_mut() else { return Ok(ScrubProgress::Idle) };
        if s.table < ALL.len() {
            // Copy the next rows of the current table from a snapshot of the live file.
            let def = ALL[s.table];
            let mut rows = Vec::new();
            let after = s.after.clone();
            self.read(|r| {
                let lo = after.clone().unwrap_or_default();
                r.scan(def, &lo, None, &mut |k, v| {
                    if after.as_deref() != Some(k) {
                        rows.push((k.to_vec(), v.to_vec()));
                    }
                    Ok(rows.len() < budget)
                })
            })?;
            let mut w = W::new(s.db.begin_write()?);
            // The fresh file is thrown away on a crash, so its copy steps needn't reach the disk; the swap's commit does.
            w.txn.set_durability(Durability::None)?;
            for (k, v) in &rows {
                w.put(def, k, v)?;
            }
            w.txn.commit()?;
            s.bytes += w.bytes;
            s.copied += rows.len() as u64;
            if rows.len() < budget {
                s.table += 1;
                s.after = None;
            } else {
                s.after = rows.last().map(|(k, _)| k.clone());
            }
            self.drain_touched(s, budget)?;
            crash_point("copying");
            return Ok(ScrubProgress::Copying { copied: s.copied });
        }
        if !s.flushed {
            // One durable commit outside every lock writes the whole copy to disk, so the swap's commit, which
            // excludes every other call, flushes only its own few keys.
            let txn = s.db.begin_write()?;
            txn.commit()?;
            s.flushed = true;
            crash_point("flushed");
            return Ok(ScrubProgress::Copying { copied: s.copied });
        }
        // Carry over what was written since the copy began, a budget at a time, outside every lock. A step that
        // carried everything waiting goes straight on to the swap, so a steady writer (whose writes between steps
        // could always outnumber SWAP_DELTA) can't hold the swap off.
        let batch: Vec<(usize, Vec<u8>)> = {
            let mut touched = self.touched.lock().unwrap_or_else(PoisonError::into_inner);
            if touched.len() > SWAP_DELTA {
                std::iter::from_fn(|| touched.pop_first()).take(budget).collect()
            } else {
                Vec::new()
            }
        };
        if !batch.is_empty() {
            let db = self.live()?;
            let from = R::new(db.begin_read()?);
            let mut w = W::new(s.db.begin_write()?);
            w.txn.set_durability(Durability::None)?;
            carry(&from, &mut w, &batch)?;
            w.txn.commit()?;
            s.bytes += w.bytes;
            s.copied += batch.len() as u64;
            if self.touched.lock().unwrap_or_else(PoisonError::into_inner).len() > SWAP_DELTA {
                return Ok(ScrubProgress::Copying { copied: s.copied });
            }
        }
        // The swap: exclude every other call for as long as it takes to carry the last few keys, commit them, mark
        // the fresh file finished, rename it, and put the fresh database in place.
        let mut live = self.db.write().unwrap_or_else(PoisonError::into_inner);
        let Some(current) = live.as_ref().map(Arc::clone) else { return Err(closed()) };
        let held = std::time::Instant::now();
        // Writes that landed since the last catch-up step are carried here too. With the lock held no more can land,
        // so if they put the swap over SWAP_DELTA, let go and catch up again rather than carry them under the lock.
        if self.touched.lock().unwrap_or_else(PoisonError::into_inner).len() > SWAP_DELTA {
            drop(live);
            let copied = guard.as_ref().map_or(0, |s| s.copied);
            return Ok(ScrubProgress::Copying { copied });
        }
        let Some(s) = guard.take() else { return Ok(ScrubProgress::Idle) };
        let rest: Vec<(usize, Vec<u8>)> = {
            let mut touched = self.touched.lock().unwrap_or_else(PoisonError::into_inner);
            self.scrubbing.store(false, Ordering::Release);
            std::mem::take(&mut *touched).into_iter().collect()
        };
        let carried = rest.len();
        let file_before = std::fs::metadata(&self.path).map_or(0, |m| m.len());
        let from = R::new(current.begin_read()?);
        drop(current);
        let mut w = W::new(s.db.begin_write()?);
        carry(&from, &mut w, &rest)?;
        let live_bytes = u64::try_from(s.bytes + w.bytes).unwrap_or(0);
        w.put(META, b"live_bytes", &live_bytes.to_be_bytes())?;
        if purge_gen(&from)? == s.started_gen {
            w.del(META, b"scrub_due")?;
        }
        // One ahead of the live file's: how many swaps this file's history has had.
        w.put(META, b"swap_count", &(swap_count(&from)? + 1).to_be_bytes())?;
        w.txn.commit()?;
        drop(from);
        crash_point("committed");
        // From the marker on, opening the catalog puts the fresh file in place. Should the marker or rename fail,
        // the fresh file goes before the lock is released, or a later start would put it over newer writes.
        let done = Self::done_path(&self.path);
        let swapped = std::fs::File::create(&done)
            .and_then(|f| f.sync_all())
            .map_err(io_error)
            .and_then(|()| sync_dir(&self.path))
            .and_then(|()| {
                crash_point("marked");
                std::fs::rename(&s.tmp, &self.path).map_err(io_error)
            });
        if let Err(e) = swapped {
            let _ = std::fs::remove_file(&s.tmp);
            let _ = std::fs::remove_file(&done);
            let _ = sync_dir(&self.path);
            return Err(e);
        }
        crash_point("renamed");
        let old = live.replace(Arc::new(s.db));
        *self.last_swap.lock().unwrap_or_else(PoisonError::into_inner) = Some((held.elapsed(), carried));
        drop(live);
        drop(guard);
        // After the lock: the directory flush makes the rename durable, then the marker can go. Until the flush
        // lands, a power loss is recovered by open(). The old file closes once no reader holds it.
        // Should the flush fail, the marker stays, which is what open() needs if the rename was lost.
        if sync_dir(&self.path).is_ok() {
            crash_point("synced");
            let _ = std::fs::remove_file(&done);
        }
        drop(old);
        let file_after = std::fs::metadata(&self.path).map_or(0, |m| m.len());
        let last = LastScrub { at_ns: now_ns(), file_before, file_after, live_bytes, rows: s.copied + carried as u64 };
        // Physical scrub statistics do not change availability. Preserve a projection carried through
        // the swap; normal writes (including the earlier scrub request) still advance its revision.
        self.write_restart(false, |w| {
            if live_bytes >= LastScrub::DENSITY_MIN_LIVE {
                w.put(META, b"density_scrub", &last.encode())?;
            }
            w.put(META, b"last_scrub", &last.encode())
        })?;
        Ok(ScrubProgress::Swapped)
    }

    fn abort_scrub(&self) -> Result<bool> {
        let mut guard = self.scrub.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(s) = guard.take() else { return Ok(false) };
        self.scrubbing.store(false, Ordering::Release);
        self.touched.lock().unwrap_or_else(PoisonError::into_inner).clear();
        drop(s.db);
        match std::fs::remove_file(&s.tmp) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(io_error(e)),
        }
        Ok(true)
    }

    fn close(&self, wait: std::time::Duration) -> Result<()> {
        // A scrub's fresh file goes first; the scrubber's thread is stopped before this, so no step races it.
        self.abort_scrub()?;
        // Every commit is durable before the file closes, whatever redb's own close does.
        self.flush()?;
        let deadline = std::time::Instant::now() + wait;
        loop {
            {
                let mut live = self.db.write().unwrap_or_else(PoisonError::into_inner);
                // No write is in flight while this is held; a reader may still hold the database for a snapshot.
                let Some(db) = live.take() else { return Ok(()) };
                match Arc::try_unwrap(db) {
                    Ok(db) => {
                        drop(live);
                        // redb's close: it records the allocator state in a durable commit and clears the header's
                        // recovery flag, so the next open loads that state instead of repairing the whole file.
                        drop(db);
                        return Ok(());
                    }
                    Err(db) => *live = Some(db),
                }
            }
            if std::time::Instant::now() >= deadline {
                return Err(CatalogError::Storage(
                    format!("a read still held the catalog after {wait:?}; it is left as a crash would leave it")
                        .into(),
                ));
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    fn storage(&self) -> Result<Option<CatalogStorage>> {
        let meta = std::fs::metadata(&self.path).map_err(io_error)?;
        let (live_bytes, last_scrub, density_scrub) = self.read(|r| {
            let live = r.get(META, b"live_bytes")?.map(|b| codec::u64_at(&b, 0)).transpose()?;
            let last = r.get(META, b"last_scrub")?.and_then(|b| LastScrub::decode(&b));
            let density = r.get(META, b"density_scrub")?.and_then(|b| LastScrub::decode(&b));
            Ok((live, last, density))
        })?;
        let scrub_file_bytes = self
            .scrub
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .map(|s| std::fs::metadata(&s.tmp).map_or(0, |m| m.len()));
        Ok(Some(CatalogStorage {
            path: self.path.clone(),
            file_bytes: meta.len(),
            disk_bytes: disk_bytes(&meta),
            live_bytes,
            scrub_file_bytes,
            last_scrub,
            density_scrub,
        }))
    }

    fn set_cursor(&self, name: &str, position: &[u8]) -> Result<()> {
        self.write(|w| w.put(CURSORS, name.as_bytes(), position))
    }

    fn cursor(&self, name: &str) -> Result<Option<Vec<u8>>> {
        self.read(|r| r.get(CURSORS, name.as_bytes()))
    }

    fn record_clock(&self, boot: BootId, point: ClockPoint) -> Result<()> {
        self.write(|w| {
            w.put(CLOCK_MAP, &Out::new().raw(&boot.0).u64(point.mono_ns).done(), &point.utc_ns.to_be_bytes())
        })
    }

    fn clock_near(&self, boot: BootId, mono_ns: u64) -> Result<ClockNear> {
        self.read(|r| {
            let point = |k: &[u8], v: &[u8]| -> Result<ClockPoint> {
                Ok(ClockPoint { mono_ns: codec::u64_at(k, 16)?, utc_ns: codec::u64_at(v, 0)? })
            };
            let key = |m: u64| Out::new().raw(&boot.0).u64(m).done();
            let at_or_before = match mono_ns.checked_add(1) {
                Some(m) => r.last_below(CLOCK_MAP, &boot.0, &key(m))?,
                None => {
                    let mut last = None;
                    r.prefix(CLOCK_MAP, &boot.0, |k, v| {
                        last = Some((k.to_vec(), v.to_vec()));
                        Ok(true)
                    })?;
                    last
                }
            };
            let at_or_before = at_or_before.map(|(k, v)| point(&k, &v)).transpose()?;
            let mut after = None;
            if let Some(m) = mono_ns.checked_add(1) {
                let hi = prefix_end(&boot.0);
                r.scan(CLOCK_MAP, &key(m), hi.as_deref(), &mut |k, v| {
                    after = Some(point(k, v)?);
                    Ok(false)
                })?;
            }
            Ok(ClockNear { at_or_before, after })
        })
    }

    fn save_cost_models(&self, bytes: &[u8]) -> Result<()> {
        self.write(|w| w.put(META, b"cost_models", bytes))
    }

    fn load_cost_models(&self) -> Result<Option<Vec<u8>>> {
        self.read(|r| r.get(META, b"cost_models"))
    }

    fn forget(&self, name: ForgottenHash) -> Result<()> {
        self.write(|w| w.put(FORGOTTEN, &name.0, &[]))
    }

    fn save_worker_state(&self, state: &WorkerState) -> Result<()> {
        worker_state::check(state)?;
        let worker = worker_state::key(&state.owner, &state.contract);
        self.write(|w| {
            let mut ids = Vec::with_capacity(state.inputs.len());
            for (name, revision) in &state.inputs {
                let Some(id) = w.get(STREAM_NAMES, name.as_bytes())? else {
                    return Err(CatalogError::Conflict(format!("input stream {name} isn't registered")));
                };
                let id = StreamId(codec::u32_at(&id, 0)?);
                if !unforgotten_since(w, id, *revision)? || !worker_state::usable(*revision) {
                    return Err(CatalogError::Conflict(format!(
                        "input stream {name} changed since the state was read"
                    )));
                }
                ids.push(id);
            }
            drop_worker_state(w, &worker)?;
            for id in &ids {
                w.put(WORKER_STATE_INPUTS, &worker_state::input_key(*id, &worker), &[])?;
            }
            w.put(WORKER_STATE, &worker, &worker_state::encode(state, &ids))
        })
    }

    fn load_worker_state(&self, owner: &str, contract: &str) -> Result<Option<WorkerState>> {
        worker_state::check_key(owner, contract)?;
        let worker = worker_state::key(owner, contract);
        self.read(|r| {
            let Some(value) = r.get(WORKER_STATE, &worker)? else {
                return Ok(None);
            };
            let row = worker_state::decode(owner, contract, &value)?;
            for (id, (name, revision)) in row.ids.iter().zip(&row.state.inputs) {
                let named = r.get(STREAM_NAMES, name.as_bytes())?.map(|v| codec::u32_at(&v, 0)).transpose()?;
                if named != Some(id.0) || !unforgotten_since(r, *id, *revision)? || !worker_state::usable(*revision) {
                    return Ok(None);
                }
            }
            Ok(Some(row.state))
        })
    }

    fn forget_worker_state(&self, owner: &str, contract: &str) -> Result<bool> {
        worker_state::check_key(owner, contract)?;
        let worker = worker_state::key(owner, contract);
        self.write(|w| drop_worker_state(w, &worker))
    }

    fn list_forgotten(&self, after: Option<ForgottenHash>, limit: usize) -> Result<ForgottenPage> {
        if !(1..=MAX_FORGOTTEN_PAGE).contains(&limit) {
            return Err(CatalogError::Invalid("forgotten page limit must be 1..=1024".into()));
        }
        self.read(|r| {
            let mut hashes = Vec::new();
            let lower = after.map_or([0; 32], |hash| hash.0);
            r.scan(FORGOTTEN, &lower, None, &mut |key, _| {
                let bytes: [u8; 32] =
                    key.try_into().map_err(|_| CatalogError::Invalid("malformed forgotten hash".into()))?;
                let hash = ForgottenHash(bytes);
                if after.is_none_or(|cursor| hash > cursor) {
                    hashes.push(hash);
                }
                Ok(hashes.len() <= limit)
            })?;
            Ok(forgotten_page(hashes, limit))
        })
    }

    fn is_forgotten(&self, name: ForgottenHash) -> Result<bool> {
        self.read(|r| r.has(FORGOTTEN, &name.0))
    }
}

#[cfg(test)]
mod window_tests {
    use super::*;
    use crate::store::{Visit, index_of};
    use continuum_core::{DomainLayer, HeaderSpec, PayloadFormat};
    use std::cell::Cell;

    struct Counted<'a> {
        inner: &'a R,
        visited: Cell<usize>,
        header_reads: Cell<usize>,
    }
    impl Reader for Counted<'_> {
        fn get(&self, def: Def, key: &[u8]) -> Result<Option<Vec<u8>>> {
            if index_of(def) == index_of(PAYLOADS) {
                self.header_reads.set(self.header_reads.get() + 1);
            }
            self.inner.get(def, key)
        }
        fn scan(&self, def: Def, lo: &[u8], hi: Option<&[u8]>, f: Visit<'_>) -> Result<()> {
            self.inner.scan(def, lo, hi, &mut |k, v| {
                if index_of(def) == index_of(TIME_INDEX) || index_of(def) == index_of(STREAM_SEQ) {
                    self.visited.set(self.visited.get() + 1);
                }
                f(k, v)
            })
        }
        fn last_below(&self, def: Def, lo: &[u8], below: &[u8]) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
            self.inner.last_below(def, lo, below)
        }
    }

    fn info() -> StreamInfo {
        StreamInfo {
            name: "fixture".into(),
            kind: StreamKind::Sensor,
            layer: DomainLayer::Layer1Raw,
            format: PayloadFormat::RawBytes,
            shape_name: "fixture".into(),
            shape_version: 1,
            rate_hz: None,
            anchor: None,
        }
    }
    fn iv(a: u64, b: u64) -> Interval {
        Interval::new(a, b).unwrap()
    }
    fn payload(n: u32, stream: StreamId, start: u64, end: u64) -> NewPayload {
        let mut bytes = [0; 16];
        bytes[12..].copy_from_slice(&n.to_be_bytes());
        let mut spec = HeaderSpec::raw(Uuid(bytes), stream, start, end);
        spec.seq = n;
        NewPayload { header: HeaderBuf::build(&spec).unwrap(), location: None, coverage: None }
    }

    #[test]
    fn time_range_cursor_keeps_reach_ties_and_maximum_sequence_boundaries() {
        let temp = tempfile::tempdir().unwrap();
        let c = RedbCatalog::open(temp.path().join("c.redb")).unwrap();
        let stream = c.register_stream(info()).unwrap();
        let max = payload(u32::MAX, stream, 100, 101);
        let mut spec = HeaderSpec::raw(Uuid([9; 16]), stream, u64::MAX - 1, u64::MAX);
        spec.seq = u32::MAX;
        let end = NewPayload { header: HeaderBuf::build(&spec).unwrap(), location: None, coverage: None };
        // A long earlier span still overlaps a later window and must survive its first page.
        c.insert(vec![
            payload(1, stream, 0, 200),
            payload(2, stream, 100, 101),
            max,
            payload(3, stream, 101, 102),
            end,
        ])
        .unwrap();
        let window = iv(100, 102);
        let first = c.range(stream, window, None, 1).unwrap();
        assert_eq!(first.items[0].header.header().seq(), 1);
        let second = c.range(stream, window, first.next, 1).unwrap();
        assert_eq!(second.items[0].header.header().seq(), 2);
        let third = c.range(stream, window, second.next, 1).unwrap();
        assert_eq!(third.items[0].header.header().seq(), u32::MAX);
        let fourth = c.range(stream, window, third.next, 1).unwrap();
        assert_eq!(fourth.items[0].header.header().seq(), 3);
        assert!(fourth.next.is_none());
        for cursor in [RangeCursor { start_ns: 102, seq: 0 }, RangeCursor { start_ns: u64::MAX, seq: u32::MAX }] {
            assert!(c.range(stream, window, Some(cursor), 1).unwrap().items.is_empty());
        }
        let last =
            c.range(stream, iv(u64::MAX - 1, u64::MAX), Some(RangeCursor { start_ns: 100, seq: u32::MAX }), 1).unwrap();
        assert_eq!(last.items.len(), 1);
        assert_eq!(last.items[0].header.header().start_ns(), u64::MAX - 1);
        assert!(
            c.range(stream, iv(0, u64::MAX), Some(RangeCursor { start_ns: u64::MAX - 1, seq: u32::MAX }), 1)
                .unwrap()
                .items
                .is_empty()
        );
    }

    #[test]
    fn time_range_pages_seek_forward_with_linear_row_visits() {
        let mut visits = Vec::new();
        for count in [96u32, 192] {
            let temp = tempfile::tempdir().unwrap();
            let c = RedbCatalog::open(temp.path().join("c.redb")).unwrap();
            let stream = c.register_stream(info()).unwrap();
            // Three distinct sequence keys at each time, so seeking must retain same-time ties.
            c.insert(
                (0..count).map(|n| payload(n + 1, stream, 100 + u64::from(n / 3), 101 + u64::from(n / 3))).collect(),
            )
            .unwrap();
            c.read(|r| {
                let counted = Counted { inner: r, visited: Cell::new(0), header_reads: Cell::new(0) };
                let mut after = None;
                let mut ids = Vec::new();
                loop {
                    let page = range_page(&counted, stream, iv(0, u64::MAX), after, 3)?;
                    ids.extend(page.items.iter().map(|p| p.header.id()));
                    match page.next {
                        Some(next) => after = Some(next),
                        None => break,
                    }
                }
                assert_eq!(ids.len(), count as usize);
                assert_eq!(ids.iter().copied().collect::<BTreeSet<_>>().len(), count as usize);
                visits.push(counted.visited.get());
                assert!(
                    counted.visited.get() <= count as usize * 2,
                    "each page must seek past its cursor, not rescan: {} visits for {count} rows",
                    counted.visited.get()
                );
                Ok(())
            })
            .unwrap();
        }
        assert!(visits[1] <= visits[0] * 2 + 2, "doubling rows must not quadruple visits: {visits:?}");
    }

    #[test]
    fn dense_window_pages_visit_each_index_row_once() {
        let temp = tempfile::tempdir().unwrap();
        let c = RedbCatalog::open(temp.path().join("c.redb")).unwrap();
        let stream = c.register_stream(info()).unwrap();
        for n in 0..32 {
            c.insert(vec![payload(n + 1, stream, 100 + (31 - n) as u64, 101 + (31 - n) as u64)]).unwrap();
        }
        c.read(|r| {
            let counted = Counted { inner: r, visited: Cell::new(0), header_reads: Cell::new(0) };
            let cache = Mutex::new(None);
            let filter = QueryFilter { window: Some(iv(100, 140)), ..QueryFilter::default() };
            let mut after = 0;
            let mut got = Vec::new();
            loop {
                let page = windowed(&counted, &cache, stream, 1, after, &filter, 3)?;
                got.extend(page.items.iter().map(|p| p.header.id()));
                match page.next_after {
                    Some(next) => {
                        assert!(next > after);
                        after = next
                    }
                    None => break,
                }
            }
            assert_eq!(got.len(), 32);
            assert_eq!(counted.visited.get(), 32, "paging must not rescan the time window");
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn window_pages_validate_each_candidate_header_once() {
        for fallback in [false, true] {
            let temp = tempfile::tempdir().unwrap();
            let c = RedbCatalog::open(temp.path().join("c.redb")).unwrap();
            let stream = c.register_stream(info()).unwrap();
            c.insert((1..=4).map(|n| payload(n, stream, 105 - u64::from(n), 106 - u64::from(n))).collect()).unwrap();
            c.read(|r| {
                let counted = Counted { inner: r, visited: Cell::new(0), header_reads: Cell::new(0) };
                let filter = QueryFilter { window: Some(iv(100, 110)), ..QueryFilter::default() };
                let mut index = WindowIndex::build(&counted, stream, iv(100, 110), c.head_seq()?, 1)?;
                if fallback {
                    index.sequences = None;
                }
                counted.header_reads.set(0);
                let first = index.page(&counted, 0, &filter, 3)?;
                assert_eq!(first.items.iter().map(|p| p.header.seq()).collect::<Vec<_>>(), [1, 2, 3]);
                assert_eq!(first.next_after, Some(3));
                assert_eq!(counted.header_reads.get(), 4, "three outputs plus one continuation candidate");
                counted.header_reads.set(0);
                let last = index.page(&counted, 3, &filter, 3)?;
                assert_eq!(last.items.iter().map(|p| p.header.seq()).collect::<Vec<_>>(), [4]);
                assert_eq!(last.next_after, None);
                assert_eq!(counted.header_reads.get(), 1, "the last header is not fetched or verified twice");
                Ok(())
            })
            .unwrap();
        }
    }

    #[test]
    fn oversized_window_pages_still_scan_linearly_with_bounded_memory() {
        let temp = tempfile::tempdir().unwrap();
        let c = RedbCatalog::open(temp.path().join("c.redb")).unwrap();
        let stream = c.register_stream(info()).unwrap();
        for n in 0..96 {
            c.insert(vec![payload(n + 1, stream, 100 + n as u64, 101 + n as u64)]).unwrap();
        }
        c.read(|r| {
            let counted = Counted { inner: r, visited: Cell::new(0), header_reads: Cell::new(0) };
            let cache = Mutex::new(None);
            let filter = QueryFilter { window: Some(iv(100, 200)), ..QueryFilter::default() };
            let mut after = 0;
            let mut got = 0;
            loop {
                let page = windowed(&counted, &cache, stream, 1, after, &filter, 3)?;
                got += page.items.len();
                match page.next_after {
                    Some(next) => after = next,
                    None => break,
                }
            }
            assert_eq!(got, 96);
            assert!(cache.lock().unwrap().as_ref().unwrap().sequences.is_none());
            assert!(counted.visited.get() <= 96 * 3, "the fallback must advance its sequence cursor");
            Ok(())
        })
        .unwrap();
    }
    #[test]
    fn newer_purged_cache_does_not_change_an_older_same_head_snapshot() {
        let temp = tempfile::tempdir().unwrap();
        let c = RedbCatalog::open(temp.path().join("c.redb")).unwrap();
        let stream = c.register_stream(info()).unwrap();
        let p = payload(1, stream, 100, 101);
        let id = p.header.id();
        c.insert(vec![p]).unwrap();
        let head = c.head_seq().unwrap();
        let slot = Mutex::new(None);
        let filter = QueryFilter { window: Some(iv(100, 110)), ..QueryFilter::default() };
        c.read(|old| {
            c.purge(PurgeSelector::Ids(vec![id]), "fixture", "", Uuid([9; 16]), 1)?;
            assert_eq!(c.head_seq()?, head);
            c.read(|new| {
                assert!(windowed(new, &slot, stream, 2, 0, &filter, 5)?.items.is_empty());
                Ok(())
            })?;
            let page = windowed(old, &slot, stream, 1, 0, &filter, 5)?;
            assert_eq!(page.items.iter().map(|p| p.header.id()).collect::<Vec<_>>(), vec![id]);
            assert_eq!(slot.lock().unwrap().as_ref().unwrap().snapshot, 2, "old read cannot replace a newer cache");
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn cached_rows_still_follow_purges_and_late_imported_commits() {
        let temp = tempfile::tempdir().unwrap();
        let c = RedbCatalog::open(temp.path().join("c.redb")).unwrap();
        let stream = c.register_stream(info()).unwrap();
        let p = payload(1, stream, 105, 106);
        let id = p.header.id();
        c.insert(vec![p]).unwrap();
        let filter = QueryFilter { window: Some(iv(100, 110)), ..QueryFilter::default() };
        assert_eq!(c.query(stream, 0, &filter, 1, 1).unwrap().items.len(), 1);
        c.purge(PurgeSelector::Ids(vec![id]), "fixture", "", Uuid([9; 16]), 1).unwrap();
        assert!(c.query(stream, 0, &filter, 1, 1).unwrap().items.is_empty());
        let late = payload(2, stream, 101, 102);
        let late_id = late.header.id();
        c.insert(vec![payload(3, stream, 900, 901), late]).unwrap();
        let page = c.query(stream, 0, &filter, 1, 1).unwrap();
        assert_eq!(page.items.iter().map(|p| p.header.id()).collect::<Vec<_>>(), vec![late_id]);
    }

    #[test]
    fn a_failed_incremental_scan_cannot_duplicate_candidates_on_retry() {
        struct Failing<'a>(&'a R);
        impl Reader for Failing<'_> {
            fn get(&self, def: Def, key: &[u8]) -> Result<Option<Vec<u8>>> {
                self.0.get(def, key)
            }
            fn scan(&self, def: Def, lo: &[u8], hi: Option<&[u8]>, f: Visit<'_>) -> Result<()> {
                self.0.scan(def, lo, hi, &mut |k, v| {
                    let keep = f(k, v)?;
                    if index_of(def) == index_of(STREAM_SEQ) {
                        return Err(CatalogError::Invalid("injected scan failure".into()));
                    }
                    Ok(keep)
                })
            }
            fn last_below(&self, def: Def, lo: &[u8], below: &[u8]) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
                self.0.last_below(def, lo, below)
            }
        }
        let temp = tempfile::tempdir().unwrap();
        let c = RedbCatalog::open(temp.path().join("c.redb")).unwrap();
        let stream = c.register_stream(info()).unwrap();
        c.insert(vec![payload(1, stream, 100, 101)]).unwrap();
        let slot = Mutex::new(None);
        let filter = QueryFilter { window: Some(iv(100, 110)), ..QueryFilter::default() };
        c.read(|r| {
            windowed(r, &slot, stream, 1, 0, &filter, 5)?;
            Ok(())
        })
        .unwrap();
        c.insert(vec![payload(2, stream, 101, 102), payload(3, stream, 102, 103)]).unwrap();
        c.read(|r| {
            assert!(windowed(&Failing(r), &slot, stream, 2, 0, &filter, 5).is_err());
            assert!(slot.lock().unwrap().is_none());
            let page = windowed(r, &slot, stream, 3, 0, &filter, 5)?;
            assert_eq!(page.items.len(), 3);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn cache_identity_separates_databases_with_the_same_commit_head() {
        let temp = tempfile::tempdir().unwrap();
        let first = RedbCatalog::open(temp.path().join("first.redb")).unwrap();
        let second = RedbCatalog::open(temp.path().join("second.redb")).unwrap();
        let stream = first.register_stream(info()).unwrap();
        assert_eq!(second.register_stream(info()).unwrap(), stream);
        first.insert(vec![payload(1, stream, 100, 101), payload(2, stream, 900, 901)]).unwrap();
        second.insert(vec![payload(3, stream, 900, 901), payload(4, stream, 100, 101)]).unwrap();
        assert_eq!(first.head_seq().unwrap(), second.head_seq().unwrap());
        let mut cache = WindowCache::default();
        let filter = QueryFilter { window: Some(iv(100, 110)), ..QueryFilter::default() };
        for (catalog, wanted, snapshot) in [(&first, 1, 1), (&second, 4, 2)] {
            let db = catalog.live().unwrap();
            let slot = cache.slot(&db, stream, iv(100, 110));
            catalog
                .read(|r| {
                    let page = windowed(r, &slot, stream, snapshot, 0, &filter, 5)?;
                    assert_eq!(page.items.len(), 1);
                    assert_eq!(page.items[0].header.id(), payload(wanted, stream, 100, 101).header.id());
                    Ok(())
                })
                .unwrap();
        }
        assert_eq!(cache.entries.len(), 2);
    }

    #[test]
    fn a_cut_frame_replaced_in_the_time_index_does_not_survive_in_a_warm_window() {
        for fallback in [false, true] {
            let temp = tempfile::tempdir().unwrap();
            let c = RedbCatalog::open(temp.path().join("c.redb")).unwrap();
            let stream = c.register_stream(info()).unwrap();
            let raw = Uuid([1; 16]);
            let chunk = Uuid([2; 16]);
            c.insert(vec![NewPayload {
                header: HeaderBuf::build(&HeaderSpec::raw(raw, stream, 100, 100)).unwrap(),
                location: Some(StorageLocator::Shm {
                    arena: "fixture".into(),
                    arena_generation: 1,
                    slab_class: 0,
                    slot: 1,
                    slot_generation: 1,
                    offset: 0,
                    length: 300,
                }),
                coverage: None,
            }])
            .unwrap();
            let filter = QueryFilter { window: Some(iv(100, 110)), ..QueryFilter::default() };
            assert_eq!(c.query(stream, 0, &filter, 5, 1).unwrap().items[0].header.id(), raw);
            if fallback {
                let slots = c.windows.lock().unwrap();
                slots.entries[0].index.lock().unwrap().as_mut().unwrap().sequences = None;
            }
            let mut cut = ChunkFrame::new(100, 0, 300);
            cut.cut = true;
            c.commit_chunk(
                NewChunk {
                    payload: NewPayload {
                        header: HeaderBuf::build(&HeaderSpec::raw(chunk, stream, 100, 101)).unwrap(),
                        location: Some(StorageLocator::Chunk {
                            tier: StorageTier::Tier1Warm,
                            uri: "file:///fixture".into(),
                            offset: 0,
                            length: 1000,
                        }),
                        coverage: None,
                    },
                    frames: vec![cut],
                },
                &[raw],
            )
            .unwrap();
            let warm = c.query(stream, 0, &filter, 5, 1).unwrap();
            assert_eq!(warm.items.iter().map(|p| p.header.id()).collect::<Vec<_>>(), vec![chunk]);
            assert_eq!(c.range(stream, iv(100, 110), None, 5).unwrap().items, warm.items);
        }
    }

    proptest::proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(32))]
        #[test]
        fn window_pagination_matches_memory_oracle(
            rows in proptest::collection::vec((0u64..100,0u64..30,proptest::bool::ANY),1..25),
            start in 0u64..100, width in 1u64..30, limit in 1usize..6,
            before in proptest::option::of(0u64..130),
            selected in proptest::option::of(proptest::collection::vec(0usize..25,0..12)),
            subjects in proptest::bool::ANY,
        ) {
            let temp=tempfile::tempdir().unwrap();let disk=RedbCatalog::open(temp.path().join("c.redb")).unwrap();
            let memory=crate::MemoryCatalog::new();let stream=disk.register_stream(info()).unwrap();
            assert_eq!(memory.register_stream(info()).unwrap(),stream);
            let want=Uuid([7;16]);let mut ids=Vec::new();
            for (n,(at,duration,about)) in rows.iter().enumerate() {
                let mut p=payload(n as u32+1,stream,*at,*at+*duration);
                let mut spec=HeaderSpec::raw(p.header.id(),stream,*at,*at+*duration);spec.seq=n as u32;
                if *about {spec.subjects=vec![want];}
                p.header=HeaderBuf::build(&spec).unwrap();ids.push(p.header.id());
                memory.insert(vec![p.clone()]).unwrap();disk.insert(vec![p]).unwrap();
            }
            let filter=QueryFilter{window:Some(iv(start,start+width)),start_before:before,
                ids:selected.map(|which|which.into_iter().filter_map(|n|ids.get(n).copied()).collect()),
                subjects:subjects.then_some(vec![want])};
            let mut after=0;
            loop {
                let actual=disk.query(stream,after,&filter,limit,1).unwrap();
                let expected=memory.query(stream,after,&filter,limit,1).unwrap();
                proptest::prop_assert_eq!(&actual,&expected);
                match actual.next_after {Some(next)=>{proptest::prop_assert!(next>after);after=next},None=>break}
            }
        }
    }
}

#[cfg(test)]
mod scrub_tests {
    use super::*;
    use continuum_core::{DomainLayer, HeaderSpec, PayloadFormat};
    use redb::TableHandle;

    const NOBODY: &[Uuid] = &[];

    fn info(name: &str) -> StreamInfo {
        StreamInfo {
            name: name.into(),
            kind: StreamKind::Sensor,
            layer: DomainLayer::Layer1Raw,
            format: PayloadFormat::RawBytes,
            shape_name: "fixture".into(),
            shape_version: 1,
            rate_hz: None,
            anchor: None,
        }
    }

    fn id(n: u32) -> Uuid {
        let mut bytes = [0; 16];
        bytes[12..].copy_from_slice(&n.to_be_bytes());
        Uuid(bytes)
    }

    fn payload(n: u32, stream: StreamId, subjects: &[Uuid]) -> NewPayload {
        let mut spec = HeaderSpec::raw(id(n), stream, u64::from(n), u64::from(n) + 1);
        spec.seq = n;
        spec.subjects = subjects.to_vec();
        NewPayload { header: HeaderBuf::build(&spec).unwrap(), location: None, coverage: None }
    }

    /// Every key and value byte outside `META`, counted the slow way.
    fn recount(c: &RedbCatalog) -> u64 {
        c.read(|r| {
            let mut total = 0u64;
            for &def in ALL {
                if def.name() == META.name() {
                    continue;
                }
                r.scan(def, &[], None, &mut |k, v| {
                    total += (k.len() + v.len()) as u64;
                    Ok(true)
                })?;
            }
            Ok(total)
        })
        .unwrap()
    }

    fn scrub_to_the_end(c: &RedbCatalog, budget: usize) {
        for _ in 0..100_000 {
            if c.scrub_step(budget).unwrap() == ScrubProgress::Swapped {
                return;
            }
        }
        panic!("the scrub never finished");
    }

    fn sidecar(path: &Path, suffix: &str) -> PathBuf {
        let mut p = path.as_os_str().to_owned();
        p.push(suffix);
        PathBuf::from(p)
    }

    const CRASH_DIR: &str = "CONTINUUM_SCRUB_CRASH_DIR";
    /// Payloads in the catalog before the scrub starts.
    const BEFORE: u32 = 300;
    /// The first id written while the scrub runs, one per step.
    const DURING: u32 = 10_000;

    /// Run in a child process by `a_crash_at_any_point_of_a_scrub_loses_nothing`: fills a catalog, then scrubs it
    /// with small steps, writing a payload, making it durable, and noting its id in `acked` after every step, until the crash point
    /// named in `CONTINUUM_SCRUB_CRASH_AT` exits the process. Does nothing in an ordinary test run.
    #[test]
    fn crash_child() {
        let Ok(dir) = std::env::var(CRASH_DIR) else { return };
        let dir = PathBuf::from(dir);
        let c = RedbCatalog::open(dir.join("catalog.redb")).unwrap();
        let s = c.register_stream(info("cam0")).unwrap();
        let forget = Uuid(*b"crash-forget-me!");
        c.insert(
            (1..=BEFORE)
                .map(|n| payload(n, s, if n % 10 == 0 { std::slice::from_ref(&forget) } else { NOBODY }))
                .collect(),
        )
        .unwrap();
        c.purge(PurgeSelector::Entity(forget), "test", "", id(9_000_000), 1).unwrap();
        // Commits fsync about once a second, so a crash may lose the last second of them by design. What
        // this test pins is the scrub's own crash safety: every write made durable before the crash survives it.
        c.flush_durable().unwrap();
        let mut acked = std::fs::File::create(dir.join("acked")).unwrap();
        for n in DURING.. {
            if c.scrub_step(16).unwrap() == ScrubProgress::Swapped {
                break;
            }
            c.insert(vec![payload(n, s, &[])]).unwrap();
            c.flush_durable().unwrap();
            use std::io::Write as _;
            writeln!(acked, "{n}").unwrap();
        }
        // Past every crash point but the last (it runs before this): say so, so the parent can tell.
        std::process::exit(0);
    }

    fn crash_at(point: &str) -> (tempfile::TempDir, PathBuf, Vec<u32>) {
        let dir = tempfile::tempdir().unwrap();
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "redb_catalog::scrub_tests::crash_child", "--nocapture", "--test-threads=1"])
            .env(CRASH_DIR, dir.path())
            .env("CONTINUUM_SCRUB_CRASH_AT", point)
            .stdout(std::process::Stdio::null())
            .status()
            .unwrap();
        assert_eq!(status.code(), Some(86), "the child crashed at {point}");
        let acked = std::fs::read_to_string(dir.path().join("acked")).unwrap_or_default();
        let acked = acked.lines().map(|l| l.parse().unwrap()).collect();
        let path = dir.path().join("catalog.redb");
        (dir, path, acked)
    }

    #[test]
    fn a_crash_at_any_point_of_a_scrub_loses_nothing() {
        // In order: the fresh file made, a copy step, the copy flushed, the swap committed, the swap marked
        // finished, the fresh file renamed into place, and the directory flushed (marker not yet removed).
        for (point, swapped) in [
            ("started", false),
            ("copying", false),
            ("flushed", false),
            ("committed", false),
            ("marked", true),
            ("renamed", true),
            ("synced", true),
        ] {
            let (_dir, path, acked) = crash_at(point);
            let c = RedbCatalog::open(&path).unwrap();
            assert!(!sidecar(&path, ".scrub").exists(), "{point}: no fresh file is left");
            assert!(!sidecar(&path, ".scrub-done").exists(), "{point}: no marker is left");
            for n in (1..=BEFORE).filter(|n| n % 10 != 0) {
                assert!(c.get(id(n)).unwrap().is_some(), "{point}: payload {n} is there");
            }
            for n in (1..=BEFORE).filter(|n| n % 10 == 0) {
                assert!(c.get(id(n)).unwrap().is_none(), "{point}: purged payload {n} stays gone");
            }
            for &n in &acked {
                assert!(c.get(id(n)).unwrap().is_some(), "{point}: write {n}, made during the scrub, is there");
            }
            let storage = c.storage().unwrap().unwrap();
            assert_eq!(storage.live_bytes.is_some(), swapped, "{point}: the file in place is the fresh one");
            if swapped {
                assert_eq!(storage.live_bytes, Some(recount(&c)), "{point}: its live bytes are exact");
            } else {
                assert!(c.scrub_needed().unwrap(), "{point}: the scrub is still due");
            }
            // And the catalog carries on: a write, then a whole scrub.
            let s = c.stream_id("cam0").unwrap().unwrap();
            c.insert(vec![payload(500_000, s, &[])]).unwrap();
            c.request_scrub().unwrap();
            scrub_to_the_end(&c, 64);
            assert!(c.get(id(500_000)).unwrap().is_some());
            assert_eq!(c.storage().unwrap().unwrap().live_bytes, Some(recount(&c)), "{point}: then exact");
        }
    }

    #[test]
    fn live_bytes_are_counted_by_the_first_scrub_and_kept_exact_after() {
        let dir = tempfile::tempdir().unwrap();
        let c = RedbCatalog::open(dir.path().join("catalog.redb")).unwrap();
        let s = c.register_stream(info("cam0")).unwrap();
        let someone = Uuid(*b"someone-to-purge");
        c.insert(
            (1..=200)
                .map(|n| payload(n, s, if n % 4 == 0 { std::slice::from_ref(&someone) } else { NOBODY }))
                .collect(),
        )
        .unwrap();
        assert_eq!(c.storage().unwrap().unwrap().live_bytes, None, "not counted before a scrub");
        c.request_scrub().unwrap();
        // Writes, deletes, and a cursor overwrite while the copy runs.
        let mut n = 1_000;
        while c.scrub_step(8).unwrap() != ScrubProgress::Swapped {
            c.insert(vec![payload(n, s, &[])]).unwrap();
            c.set_cursor("sub", &n.to_be_bytes()).unwrap();
            n += 1;
        }
        assert!(n > 1_010, "the scrub took several steps");
        assert_eq!(c.storage().unwrap().unwrap().live_bytes, Some(recount(&c)), "counted at the swap");
        c.insert((2_000..2_050).map(|n| payload(n, s, &[])).collect()).unwrap();
        c.set_cursor("sub", b"a much longer cursor position than before").unwrap();
        c.purge(PurgeSelector::Entity(someone), "test", "", id(9_000), 1).unwrap();
        assert_eq!(c.storage().unwrap().unwrap().live_bytes, Some(recount(&c)), "kept by every write after");
        let last = c.storage().unwrap().unwrap().last_scrub.unwrap();
        assert!(last.at_ns > 0 && last.file_after > 0 && last.rows >= 200);
    }

    #[test]
    fn the_written_keys_kept_for_the_swap_stay_about_one_step_long() {
        let dir = tempfile::tempdir().unwrap();
        let c = RedbCatalog::open(dir.path().join("catalog.redb")).unwrap();
        let s = c.register_stream(info("cam0")).unwrap();
        c.insert((1..=2_000).map(|n| payload(n, s, &[])).collect()).unwrap();
        c.request_scrub().unwrap();
        let (mut n, mut most) = (100_000, 0);
        loop {
            // Ten payloads a step, each about seven rows: the set never holds much more than a step's writes.
            c.insert((n..n + 10).map(|n| payload(n, s, &[])).collect()).unwrap();
            n += 10;
            most = most.max(c.touched.lock().unwrap().len());
            if c.scrub_step(64).unwrap() == ScrubProgress::Swapped {
                break;
            }
        }
        assert!(n > 100_200, "the scrub took many steps");
        assert!(most <= 2 * 10 * 8 + SWAP_DELTA, "at most about two steps of writes were waiting: {most}");
        for k in 100_000..n {
            assert!(c.get(id(k)).unwrap().is_some(), "write {k} was carried");
        }
        assert_eq!(c.storage().unwrap().unwrap().live_bytes, Some(recount(&c)));
    }

    #[test]
    fn a_small_scrub_keeps_the_last_large_scrubs_density() {
        let dir = tempfile::tempdir().unwrap();
        let c = RedbCatalog::open(dir.path().join("catalog.redb")).unwrap();
        let s = c.register_stream(info("cam0")).unwrap();
        c.insert((1..=50).map(|n| payload(n, s, &[])).collect()).unwrap();
        // A file whose last scrub was small and nothing before it: no density to go by.
        c.request_scrub().unwrap();
        scrub_to_the_end(&c, 16);
        let first = c.storage().unwrap().unwrap();
        assert!(first.last_scrub.is_some_and(|l| l.live_bytes < LastScrub::DENSITY_MIN_LIVE));
        assert_eq!(first.density_scrub, None, "a scrub of a small file says nothing about density");
        // An earlier large scrub's record (as one with 2 GiB live writes it) outlives a later small one.
        let large = LastScrub { at_ns: 1, file_before: 6 << 30, file_after: 5 << 30, live_bytes: 2 << 30, rows: 9 };
        c.write(|w| w.put(META, b"density_scrub", &large.encode())).unwrap();
        c.request_scrub().unwrap();
        scrub_to_the_end(&c, 16);
        let after = c.storage().unwrap().unwrap();
        assert_ne!(after.last_scrub, first.last_scrub, "the second scrub swapped");
        assert_eq!(after.density_scrub, Some(large), "kept through the copy, not replaced by the small scrub");
    }

    #[test]
    fn an_aborted_scrub_leaves_no_fresh_file_and_stays_due() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("catalog.redb");
        let c = RedbCatalog::open(&path).unwrap();
        let s = c.register_stream(info("cam0")).unwrap();
        c.insert((1..=100).map(|n| payload(n, s, &[])).collect()).unwrap();
        assert!(!c.abort_scrub().unwrap(), "nothing to abort");
        c.request_scrub().unwrap();
        for _ in 0..3 {
            assert!(matches!(c.scrub_step(16).unwrap(), ScrubProgress::Copying { .. }));
        }
        assert!(c.storage().unwrap().unwrap().scrub_file_bytes.is_some_and(|b| b > 0));
        assert!(c.abort_scrub().unwrap());
        assert!(!sidecar(&path, ".scrub").exists());
        assert_eq!(c.storage().unwrap().unwrap().scrub_file_bytes, None);
        assert!(c.scrub_needed().unwrap(), "still due");
        c.insert(vec![payload(1_000, s, &[])]).unwrap();
        scrub_to_the_end(&c, 16);
        assert!(c.get(id(1_000)).unwrap().is_some() && c.get(id(50)).unwrap().is_some());
        assert!(!c.scrub_needed().unwrap());
    }

    #[test]
    fn a_stale_marker_is_cleared_before_a_new_fresh_file_exists() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("catalog.redb");
        let c = RedbCatalog::open(&path).unwrap();
        let s = c.register_stream(info("cam0")).unwrap();
        c.insert((1..=50).map(|n| payload(n, s, &[])).collect()).unwrap();
        // A marker whose removal after a finished swap was lost.
        std::fs::write(sidecar(&path, ".scrub-done"), b"").unwrap();
        c.request_scrub().unwrap();
        assert!(matches!(c.scrub_step(16).unwrap(), ScrubProgress::Copying { .. }));
        assert!(sidecar(&path, ".scrub").exists() && !sidecar(&path, ".scrub-done").exists());
        drop(c);
        // So a crash now throws the half-made copy away.
        let c = RedbCatalog::open(&path).unwrap();
        assert!(!sidecar(&path, ".scrub").exists() && c.get(id(50)).unwrap().is_some());
    }
}

#[cfg(test)]
mod close_tests {
    use super::*;
    use std::time::Duration;

    /// redb's god byte (header offset 9) has bit 2 set while a writer holds the file, cleared only by a clean close.
    fn recovery_required(path: &Path) -> bool {
        std::fs::read(path).unwrap()[9] & 2 != 0
    }

    #[test]
    fn closing_shuts_the_file_down_cleanly_and_refuses_later_calls() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("catalog.redb");
        let catalog = RedbCatalog::open(&path).unwrap();
        catalog.save_cost_models(b"costs").unwrap();
        assert!(recovery_required(&path), "an open catalog is marked in use");
        catalog.close(Duration::from_secs(5)).unwrap();
        assert!(!recovery_required(&path), "the next open would repair the whole file");
        assert!(catalog.load_cost_models().is_err(), "a read after the close is refused");
        assert!(catalog.save_cost_models(b"later").is_err(), "a write after the close is refused");
        catalog.close(Duration::from_secs(5)).unwrap();
        drop(catalog);
        let reopened = RedbCatalog::open(&path).unwrap();
        assert_eq!(reopened.load_cost_models().unwrap().as_deref(), Some(&b"costs"[..]));
    }

    /// Commits within DURABLE_EVERY of the last durable one skip the fsync; a flush or a clean close makes
    /// every one of them durable.
    #[test]
    fn commits_inside_the_durability_window_are_pending_until_a_flush_or_close() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("catalog.redb");
        let catalog = RedbCatalog::open(&path).unwrap();
        for n in 0..50u8 {
            catalog.save_cost_models(&[n]).unwrap(); // well inside a second of the open: none fsyncs
        }
        assert!(catalog.undurable.load(Ordering::Acquire), "the commits since the last fsync wait for one");
        catalog.flush_durable().unwrap();
        assert!(!catalog.undurable.load(Ordering::Acquire), "one empty durable commit covers them all");
        catalog.flush_durable().unwrap(); // nothing waiting: no commit at all
        catalog.save_cost_models(b"last").unwrap();
        assert!(catalog.undurable.load(Ordering::Acquire));
        catalog.close(Duration::from_secs(5)).unwrap();
        drop(catalog);
        let reopened = RedbCatalog::open(&path).unwrap();
        assert_eq!(reopened.load_cost_models().unwrap().as_deref(), Some(&b"last"[..]), "a clean close kept the last");
    }

    #[test]
    fn a_commit_a_second_after_the_last_durable_one_is_durable_itself() {
        let temp = tempfile::tempdir().unwrap();
        let catalog = RedbCatalog::open(temp.path().join("catalog.redb")).unwrap();
        *catalog.durable.lock().unwrap() = std::time::Instant::now() - DURABLE_EVERY;
        catalog.save_cost_models(b"costs").unwrap();
        assert!(!catalog.undurable.load(Ordering::Acquire), "past the window, the commit fsyncs");
        catalog.close(Duration::from_secs(5)).unwrap();
    }

    #[test]
    fn closing_waits_for_a_reader_and_gives_up_after_the_wait() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("catalog.redb");
        let catalog = RedbCatalog::open(&path).unwrap();
        let reading = catalog.live().unwrap();
        assert!(catalog.close(Duration::from_millis(20)).is_err());
        assert!(catalog.load_cost_models().is_ok(), "a close that gave up leaves the catalog open");
        drop(reading);
        catalog.close(Duration::from_secs(5)).unwrap();
        assert!(!recovery_required(&path));
    }

    #[test]
    fn closing_throws_away_a_running_scrub() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("catalog.redb");
        let catalog = RedbCatalog::open(&path).unwrap();
        catalog.write(|w| w.put(META, b"scrub_due", &[1])).unwrap();
        let fresh = RedbCatalog::scrub_path(&path);
        catalog.save_cost_models(b"costs").unwrap();
        assert!(matches!(catalog.scrub_step(1).unwrap(), ScrubProgress::Copying { .. }));
        assert!(fresh.exists());
        catalog.close(Duration::from_secs(5)).unwrap();
        assert!(!fresh.exists(), "the scrub's fresh file is deleted");
        assert!(!recovery_required(&path));
    }

    /// Breaks both commit slots' checksums, so redb refuses the file as corrupt.
    fn corrupt_both_slots(path: &Path) {
        let mut bytes = std::fs::read(path).unwrap();
        for slot in [64, 64 + 128] {
            bytes[slot + 112] ^= 0xFF;
        }
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn a_corrupt_open_leaves_a_marker_and_later_opens_leave_the_file_alone() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("catalog.redb");
        let catalog = RedbCatalog::open(&path).unwrap();
        catalog.close(Duration::from_secs(5)).unwrap();
        drop(catalog);
        corrupt_both_slots(&path);
        let first = RedbCatalog::open(&path).err().expect("a corrupt file is refused").to_string();
        assert!(first.contains("corrupted"), "{first}");
        let marker = RedbCatalog::open_failed_path(&path);
        assert!(std::fs::read_to_string(&marker).unwrap().contains("corrupted"), "the marker says why");
        let evidence = std::fs::read(&path).unwrap();
        for _ in 0..3 {
            let again = RedbCatalog::open(&path).err().expect("refused while the marker is there").to_string();
            assert!(again.contains("left as it is") && again.contains(".open-failed"), "{again}");
        }
        assert_eq!(std::fs::read(&path).unwrap(), evidence, "the refused opens never touched the file");
        // Removing the marker is what lets an open try the file again.
        std::fs::remove_file(&marker).unwrap();
        assert!(RedbCatalog::open(&path).is_err());
        assert!(marker.exists());
    }

    #[test]
    fn a_catalog_that_is_not_corrupt_leaves_no_marker_on_any_other_failure() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("catalog.redb");
        std::fs::write(&path, b"not a redb file at all").unwrap();
        assert!(RedbCatalog::open(&path).is_err());
        assert!(!RedbCatalog::open_failed_path(&path).exists(), "only redb's corruption verdict leaves the marker");
    }
}

#[cfg(test)]
mod open_timings_tests {
    use super::*;

    #[test]
    fn successful_open_timings_account_for_phases_without_changing_reopen() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("catalog.redb");
        for _ in 0..2 {
            let catalog = RedbCatalog::open(&path).unwrap();
            let timings = catalog.open_timings();
            assert!(
                timings.scrub_recovery + timings.database_open + timings.schema_setup + timings.recording_repair
                    <= timings.total
            );
            assert_eq!(catalog.open_timings(), timings, "getter is an immutable process-local snapshot");
            assert_eq!(catalog.head_seq().unwrap(), 0);
            catalog.close(std::time::Duration::from_secs(1)).unwrap();
        }
    }
}
