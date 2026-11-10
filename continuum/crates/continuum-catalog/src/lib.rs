//! The Continuum Broker's catalog (docs/CONTINUUM.md, "Catalog"): every payload's header, where its bytes are, what
//! it came from, which time ranges each stage has processed, and the purge that removes everything derived from what
//! it purges.
//!
//! [`Catalog`] is the contract the broker codes against. [`RedbCatalog`] is the broker's, in a redb file;
//! [`MemoryCatalog`] keeps everything in memory, for tests and for the broker's own tests. Both pass the same
//! conformance suite (`tests/conformance.rs`).
//!
//! Calls are synchronous: redb is synchronous with one writer, and the broker runs the catalog on its own blocking
//! thread. Nothing reads a clock: a call that records a time is given it (`now_ns`).

mod codec;
mod coverage;
mod memory;
mod provenance;
pub use provenance::{ChunkProvenance, FrameProvenance, PROVENANCE_VERSION};
mod recording;
pub use recording::{
    MAX_RECORDING_BYTES, MAX_RECORDING_ITEMS, MAX_RECORDING_SCAN, RecordingCursor, RecordingInput, RecordingPage,
};
mod restart;
pub use restart::{MAX_RESTART_HOT_IDS, MAX_RESTART_PROJECTION_BYTES, RestartMiss, RestartProjection, RestartStamp};
mod redb_catalog;
mod store;
mod tables;
mod types;
mod worker_state;
pub use worker_state::{MAX_WORKER_STATE_BYTES, MAX_WORKER_STATE_INPUTS, WorkerState};

pub use coverage::CoverageSet;
pub use memory::MemoryCatalog;
pub use redb_catalog::{InstantOriginal, OpenTimings, RedbCatalog, SWAP_DELTA};

/// The most frames one chunk may absorb, so one commit stays short. Both catalogs refuse more.
pub const MAX_CHUNK_FRAMES: usize = 4096;

// Cooperative purge batches check their caller between small, resumable chunks of existing phase work.
const PURGE_BATCH_ROWS: usize = 64;
pub use types::*;

use continuum_core::{BootId, Interval, StageId, StorageLocator, StorageTier, StreamId, Uuid};

/// The broker's metadata store. Every write is one transaction: it all lands, or none of it does.
pub trait Catalog: Send + Sync {
    /// The restart tracking token for a quiescent projection, if this catalog keeps one. A redb file has tracking
    /// from its creation.
    /// # Errors
    /// Catalog storage or integrity failure.
    fn restart_stamp(&self) -> Result<Option<RestartStamp>> {
        Ok(None)
    }

    /// Load a fresh opaque checkpoint and exact hot IDs from one catalog snapshot.
    /// Missing, stale, malformed, or over-budget checkpoint bytes return the [`RestartMiss`] that says which.
    /// # Errors
    /// Catalog storage or tracking-integrity failure.
    fn load_restart_projection(
        &self,
        _max_bytes: usize,
    ) -> Result<std::result::Result<RestartProjection, RestartMiss>> {
        Ok(Err(RestartMiss::Unsupported))
    }

    /// Publish only if the stamp still matches; publication itself does not advance the revision.
    /// # Errors
    /// Invalid byte budget, catalog storage, or tracking-integrity failure.
    fn save_restart_projection(&self, _expected: RestartStamp, _bytes: &[u8]) -> Result<bool> {
        Ok(false)
    }

    /// Original native microphone inputs and explicit unknown history, independent of current copies.
    /// All rows share one bounded time cursor. Missing legacy metadata never certifies empty history.
    /// # Errors
    /// Invalid budgets/source, unavailable revision, corrupt evidence, or a changed expected revision.
    #[allow(clippy::too_many_arguments)]
    fn recording_page(
        &self,
        stream: StreamId,
        window: Interval,
        after: Option<RecordingCursor>,
        expected_revision: Option<u64>,
        max_items: usize,
        max_scan: usize,
        max_bytes: usize,
    ) -> Result<RecordingPage> {
        let _ = (stream, window, after, expected_revision, max_items, max_scan, max_bytes);
        Err(CatalogError::Invalid("this catalog cannot prove recording applicability".into()))
    }

    /// Registers a stream, or returns the id it already has when the name is registered with the same identity: kind,
    /// layer, format, and shape. A capture worker re-registers its stream after every restart with a fresh clock
    /// anchor, so a re-registration replaces `anchor` and `rate_hz` rather than counting them.
    ///
    /// # Errors
    /// [`CatalogError::Conflict`] when the name is registered with a different identity.
    fn register_stream(&self, info: StreamInfo) -> Result<StreamId>;

    /// A stream's info.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn stream(&self, id: StreamId) -> Result<Option<StreamInfo>>;

    /// Monotone per-stream revision for read-visible removals and frame cuts, atomic with those changes.
    /// Appended commits need not advance it. `None` means this implementation cannot supply the guarantee.
    ///
    /// # Errors
    /// Supporting implementations reject unknown streams.
    fn invalidation_revision(&self, _stream: StreamId) -> Result<Option<u64>> {
        Ok(None)
    }

    /// The stream's invalidation revision as its last forget left it (#1587): every change but an age-out's. A
    /// dependency or worker state read at any revision from this one through the current one is still good, since
    /// aging isn't forgetting. A worker that only needs to know whether something it read was forgotten compares this
    /// one. `None` when unsupported or exhausted.
    ///
    /// # Errors
    /// Supporting implementations reject unknown streams.
    fn forget_revision(&self, _stream: StreamId) -> Result<Option<u64>> {
        Ok(None)
    }

    /// The stream registration generation: it goes up by one, atomically, each time a new stream is registered (never
    /// for a re-registration of a known one, unless its source anchor changed). `None` means this implementation
    /// can't supply it.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn stream_generation(&self) -> Result<Option<u64>> {
        Ok(None)
    }

    /// Registered streams in id order, after `after`, at most `limit`, for loading the broker's index at startup.
    ///
    /// # Errors
    /// [`CatalogError::Invalid`] for a `limit` of 0.
    fn streams(&self, after: Option<StreamId>, limit: usize) -> Result<Vec<(StreamId, StreamInfo)>>;

    /// Every (source stream, stage) pair that has coverage, in order, after `after`, at most `limit`.
    ///
    /// # Errors
    /// [`CatalogError::Invalid`] for a `limit` of 0.
    fn coverage_keys(&self, after: Option<(StreamId, StageId)>, limit: usize) -> Result<Vec<(StreamId, StageId)>>;

    /// A stream's id by name.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn stream_id(&self, name: &str) -> Result<Option<StreamId>>;

    /// Indexes payloads in one transaction: headers, locations, provenance edges and subjects from the headers, and
    /// any coverage they finish. A payload's parents may be earlier in the same batch. Durable when it returns.
    ///
    /// Each payload gets the next commit sequence: one counter for the whole catalog, strictly increasing and never
    /// reused, not even after a purge removes the payload that had it. It returns one [`Inserted`] per payload, in
    /// order: the sequence, and whether the payload landed hidden under a running purge.
    ///
    /// It is idempotent on payload id: a payload whose id is indexed already with identical header bytes is not
    /// written again (its location and coverage are ignored), and its existing commit sequence is returned. Once a
    /// purge has removed an id, the catalog doesn't know it any more, so inserting it again is a new payload: a
    /// producer must not retry an insert across a purge that took it.
    ///
    /// An inline location ([`StorageLocator::Inline`]) holds at most [`continuum_core::MAX_INLINE_BYTES`]; a
    /// larger payload goes in shared memory.
    ///
    /// # Errors
    /// [`CatalogError::NotFound`] for an unregistered stream, [`CatalogError::Conflict`] for an id indexed with a
    /// different header (or absorbed into a chunk) or a used (stream, start, seq), [`CatalogError::Invalid`] for an
    /// unknown parent or an inline location over the cap. Nothing is written on any error.
    fn insert(&self, payloads: Vec<NewPayload>) -> Result<Vec<Inserted>>;

    /// [`Catalog::insert`], with each payload's [`Dependency`]s (by id) stored in the same commit, so the payload is
    /// never indexed without them. A repeat keeps the dependencies it has. A dependency's stream must be registered.
    ///
    /// # Errors
    /// As [`Catalog::insert`]; [`CatalogError::NotFound`] for an unregistered stream, [`CatalogError::Invalid`] for a
    /// dependency of a payload not in `payloads`, or when this catalog keeps no dependencies.
    fn insert_dependent(
        &self,
        payloads: Vec<NewPayload>,
        dependencies: Vec<(Uuid, Vec<Dependency>)>,
    ) -> Result<Vec<Inserted>> {
        if dependencies.iter().any(|(_, d)| !d.is_empty()) {
            return Err(CatalogError::Invalid("this catalog keeps no dependencies".into()));
        }
        self.insert(payloads)
    }

    /// The last commit sequence assigned, 0 before the first. A reader that has seen this has seen everything
    /// committed so far.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn head_seq(&self) -> Result<u64>;

    /// Payloads on `stream` with a commit sequence above `after_seq`, in commit order, that `filter` keeps: at most
    /// `limit` of them, after looking at no more than `max_scan` rows, so a selective filter stays bounded too.
    /// Payloads a running purge has hidden are skipped. With `filter.window` set, the rows looked at are the window's
    /// (from the time index), and `max_scan` does not apply. redb caches window sequences across pages within a
    /// fixed memory budget; an oversized window advances through stream sequences from its first matching commit.
    ///
    /// # Errors
    /// [`CatalogError::NotFound`] for an unregistered stream, [`CatalogError::Invalid`] for a `limit` or `max_scan`
    /// of 0.
    fn query(
        &self,
        stream: StreamId,
        after_seq: u64,
        filter: &QueryFilter,
        limit: usize,
        max_scan: usize,
    ) -> Result<SeqPage>;

    /// What an id is: a payload, an absorbed frame (through `FRAMES`), or nothing.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn get(&self, id: Uuid) -> Result<Option<Resolved>>;

    /// Payloads on `stream` whose span ([`Interval::covering`]) overlaps `window`, by (`start_ns`, `seq`), starting
    /// after `after` and at most `limit` of them. A chunk that began before the window but reaches into it is
    /// included.
    ///
    /// # Errors
    /// [`CatalogError::NotFound`] for an unregistered stream, [`CatalogError::Invalid`] for a `limit` of 0.
    fn range(&self, stream: StreamId, window: Interval, after: Option<RangeCursor>, limit: usize) -> Result<Page>;

    /// The payloads `id` was computed from (one hop). A parent a running purge has hidden is still listed for a child
    /// it hasn't reached yet, while [`Catalog::descendants`] never lists what is under a hidden payload.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only. An unknown id has none.
    fn parents(&self, id: Uuid) -> Result<Vec<Uuid>>;

    /// The payloads computed from `id` (one hop), in id order.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only. An unknown id has none.
    fn children(&self, id: Uuid) -> Result<Vec<Uuid>>;

    /// Everything derived from `roots` through provenance edges, transitively, nearest first, without the roots, at
    /// most `limit`. A chunk's absorbed frames aren't derived from it, so they aren't listed.
    ///
    /// # Errors
    /// [`CatalogError::Invalid`] for a `limit` of 0.
    fn descendants(&self, roots: &[Uuid], limit: usize) -> Result<Vec<Uuid>>;

    /// Records where a copy of a payload's bytes is, replacing the one in that locator's tier. Inline bytes are set at
    /// insert and never move, so they can't be replaced or set this way.
    ///
    /// # Errors
    /// [`CatalogError::NotFound`] when the payload isn't indexed, [`CatalogError::Conflict`] when a running purge
    /// has hidden it (no new copies of what is being purged, and the purge's delta already named its copies),
    /// [`CatalogError::Invalid`] for an inline locator, or for any locator over a payload whose bytes are inline.
    fn set_location(&self, id: Uuid, locator: StorageLocator) -> Result<()>;

    /// Forgets the copy in `tier` (a Tier 0 slot freed after its flush, say).
    ///
    /// # Errors
    /// [`CatalogError::NotFound`] when the payload isn't indexed, [`CatalogError::Conflict`] when a running purge
    /// has hidden it (its delta already named its copies; the purge's notices free them), [`CatalogError::Invalid`] for
    /// inline bytes, which are never cleared this way.
    fn clear_location(&self, id: Uuid, tier: StorageTier) -> Result<()>;

    /// Forgets the copies in `tier` of many payloads at once (a restart clearing the last run's Tier 0 rows, say), in
    /// one transaction where the store has them. A payload that isn't indexed, is hidden by a running purge, or holds
    /// inline bytes in `tier` is skipped, as [`Catalog::clear_location`] would refuse it. Returns how many were
    /// cleared.
    ///
    /// # Errors
    /// A storage error, in which case none of this call's clears happened (in a transactional store).
    fn clear_locations(&self, ids: &[Uuid], tier: StorageTier) -> Result<usize> {
        let mut cleared = 0;
        for &id in ids {
            match self.clear_location(id, tier) {
                Ok(()) => cleared += 1,
                Err(CatalogError::NotFound(_) | CatalogError::Conflict(_) | CatalogError::Invalid(_)) => {}
                Err(e) => return Err(e),
            }
        }
        Ok(cleared)
    }

    /// Commits a chunk and compacts the frames it absorbed, in one transaction: the chunk becomes a payload with its
    /// frame index, and each absorbed frame's `PAYLOADS`, `TIME_INDEX`, and `LOCATIONS` rows go. Every absorbed frame
    /// keeps a `FRAMES` row (chunk, index, subjects), so its id still resolves as a parent, for a child committed
    /// after the absorption too (ASR and a VLM take seconds; the tier worker absorbs within about two), and a purge
    /// still finds it. `absorbed[i]` is `chunk.frames[i]`.
    ///
    /// Only raw frames are absorbed: a payload with parents (a copy hydrated from a chunk, say) is refused, so a
    /// chunk never holds a copy of another chunk's bytes.
    ///
    /// A frame a purge is removing, or has removed (no row left for its id), is not refused: the chunk is committed
    /// with that frame marked cut and the frame is not absorbed (the purge takes care of its row; the tier worker
    /// rewrites the chunk without the cut records when the purge's notice reaches it). The tier worker closes and
    /// commits a chunk whatever a purge did meanwhile, and the unpurged frames must still reach the catalog.
    ///
    /// # Errors
    /// What [`Catalog::insert`] returns for the chunk, and [`CatalogError::Invalid`] when: the frame index and
    /// `absorbed` differ in length; an absorbed id is the chunk's own, is indexed on another stream, or has parents;
    /// `frames[i].start_ns` isn't `absorbed[i]`'s start for a live frame, the starts go backwards, or one falls
    /// outside the chunk's span; it absorbs more than [`MAX_CHUNK_FRAMES`]; or the chunk has no location, or one in
    /// Tier 0. A live frame the worker marks cut (a purge reached the file before the commit) isn't absorbed: it
    /// stays its own payload, and the chunk lands without its slot. It returns what [`Catalog::insert`] would for the
    /// chunk, with [`Inserted::cut`] naming the frames the chunk landed without.
    fn commit_chunk(&self, chunk: NewChunk, absorbed: &[Uuid]) -> Result<Inserted>;

    /// [`Catalog::commit_chunk`], also recording what capture attested about each frame ([`ChunkProvenance`]) in
    /// the same transaction. `None` is a plain commit, and the chunk's provenance stays
    /// unknown. A retry of an identical commit keeps what the first stored. Rewrites and aging keep frames in place,
    /// so the provenance never changes after this.
    ///
    /// # Errors
    /// What [`Catalog::commit_chunk`] returns, and [`CatalogError::Invalid`] when the provenance has an unknown
    /// version or a different count than the frame index, or this catalog can't keep provenance.
    fn commit_chunk_with_provenance(
        &self,
        chunk: NewChunk,
        absorbed: &[Uuid],
        provenance: Option<&ChunkProvenance>,
    ) -> Result<Inserted> {
        match provenance {
            None => self.commit_chunk(chunk, absorbed),
            Some(_) => Err(CatalogError::Invalid("this catalog can't keep frame provenance".into())),
        }
    }

    /// A visible chunk's per-frame capture provenance, as committed. `None` when unknown (a chunk committed without
    /// one), never an empty claim of continuity.
    ///
    /// # Errors
    /// Storage or a malformed persisted row.
    fn chunk_provenance(&self, chunk: Uuid) -> Result<Option<ChunkProvenance>> {
        let _ = chunk;
        Ok(None)
    }

    /// Records a chunk the tier worker rewrote (after a purge cut frames from it, or on recompression): its new frame
    /// index and location. The frames stay the same frames, in the same places: same count, starts, and cut flags.
    ///
    /// # Errors
    /// [`CatalogError::NotFound`] when `chunk` isn't a chunk, [`CatalogError::Invalid`] when the frames differ from
    /// the chunk's in count, starts, or cut flags, or the location is in Tier 0, [`CatalogError::Conflict`] when a
    /// running purge has hidden the chunk.
    fn rewrite_chunk(&self, chunk: Uuid, frames: Vec<ChunkFrame>, location: StorageLocator) -> Result<()>;

    /// Read a visible chunk, its copies, and its provenance from one coherent catalog snapshot.
    /// Unknown or hidden IDs return None. An indexed nonchunk is Invalid, as is an unsupported implementation.
    /// # Errors
    /// Unsupported snapshots, a nonchunk ID, malformed metadata, or storage failures.
    fn chunk_snapshot(&self, chunk: Uuid) -> Result<Option<ChunkSnapshot>> {
        let _ = chunk;
        Err(CatalogError::Invalid("this catalog cannot read atomic chunk snapshots".into()))
    }

    /// Records `spans` of `stream` as aged out, merged with what is there: a retention policy removed what was
    /// recorded there, so readers can say so instead of treating the stretch as never recorded. Age-outs record their
    /// own spans; this is for anything else a policy drops.
    ///
    /// # Errors
    /// [`CatalogError::NotFound`] for an unregistered stream.
    fn record_aged_out(&self, stream: StreamId, spans: &[Interval]) -> Result<()> {
        let _ = (stream, spans);
        Err(CatalogError::Invalid("this catalog can't record aged-out spans".into()))
    }

    /// The parts of `window` on `stream` recorded as aged out, in order, the first `max` of them.
    ///
    /// # Errors
    /// [`CatalogError::Invalid`] for a `max` of 0.
    fn aged_out(&self, stream: StreamId, window: Interval, max: usize) -> Result<Vec<Interval>> {
        let _ = (stream, window, max);
        Ok(Vec::new())
    }

    /// Where the catalog file's bytes go, from one read snapshot: each table's pages and bytes and the file's
    /// apparent and on-disk size. It walks every table's pages, so call it off the writer. `None` for a catalog that
    /// keeps no file.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] when the file can't be read.
    fn storage_stats(&self) -> Result<Option<StorageStats>> {
        Ok(None)
    }

    /// Notes that everything committed so far was committed by `now_ns` (UTC): one `COMMIT_CLOCK` row of the head
    /// sequence. The broker calls it about once a minute. A time at or before the last row's is ignored, so the
    /// clock never goes back.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn mark_commit_clock(&self, now_ns: u64) -> Result<()> {
        let _ = now_ns;
        Ok(())
    }

    /// The highest commit sequence known to have been committed at or before `at_ns`, from the commit clock: 0 when
    /// no mark is that old. Ages count from this, not from a payload's own time, which an import sets in the past.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn commit_seq_at(&self, at_ns: u64) -> Result<u64> {
        let _ = at_ns;
        Ok(0)
    }

    /// Merges finished coverage into `(mark.source, mark.stage)`.
    ///
    /// # Errors
    /// [`CatalogError::NotFound`] for an unregistered source stream.
    fn add_coverage(&self, mark: CoverageMark) -> Result<()>;

    /// The covered parts of `window` for `(source, stage)`, in order.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn coverage(&self, source: StreamId, stage: StageId, window: Interval) -> Result<Vec<Interval>>;

    /// The parts of `window` that `(source, stage)` hasn't processed, in order, the first `max` of them.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn coverage_gaps(&self, source: StreamId, stage: StageId, window: Interval, max: usize) -> Result<Vec<Interval>>;

    /// `reason` and `detail` (the caller's JSON) are kept in the purge's record for good, and a scrub copies them, so
    /// `detail` must not name a forgotten person: no subject id.
    ///
    /// Starts purging what `selector` names and everything derived from it, transitively, as a job of bounded
    /// steps ([`Catalog::purge_step`]), so ingest commits keep landing between them. It writes the `PURGES` record
    /// under `purge_id` and returns at once. `FORGOTTEN` is never touched.
    ///
    /// Its steps find what to purge and hide it from every read as they go, each step's delta saying what it hid,
    /// and a payload inserted later with a parent the purge has hidden is hidden and purged with it, so nothing
    /// about a forgotten person is served or derived while the purge runs.
    ///
    /// A frame absorbed into a chunk that stays is cut: marked in the chunk's frame index from the step that finds
    /// it (readers skip cut frames), named in that step's delta, and listed in the report's partial chunks for the
    /// tier worker. Its time is its slot in the chunk, `[its start, the next frame's start)`
    /// (to the chunk's end for the last frame). Whatever was derived from the chunk over that slot goes too, since a
    /// copy hydrated from a chunk names the chunk, not the frame, as its parent. That includes something derived
    /// from the whole chunk: a purge removes too much rather than leave a copy.
    ///
    /// Every stage's coverage loses the spans of the payloads purged from its source stream, the cut frames' slots,
    /// and for a range the range itself on the purged streams, so nothing counts as done over data that is gone.
    ///
    /// # Errors
    /// [`CatalogError::Conflict`] when `purge_id` is used, [`CatalogError::NotFound`] for an unregistered stream in a
    /// range selector.
    fn start_purge(
        &self,
        selector: PurgeSelector,
        reason: &str,
        detail: &str,
        purge_id: Uuid,
        now_ns: u64,
    ) -> Result<()>;

    /// A durable catalog keeps a running purge's job (its selector, what it has hidden, and what is left to do) until
    /// the report is written, so after a restart [`Catalog::pending_purges`] lists it and its steps carry on; what it
    /// had hidden stays hidden.
    ///
    /// Runs one step of a purge, handling at most about `budget` rows in one short write, and says what it changed
    /// (a [`PurgeDelta`]: what it hid, frames it cut, and coverage it took out), so the broker's availability index
    /// can follow in the same step. Call it until it returns [`PurgeProgress::Removed`]; calling it after that
    /// returns the same report again, with an empty delta.
    ///
    /// # Errors
    /// [`CatalogError::NotFound`] when there is no such purge, [`CatalogError::Invalid`] for a `budget` of 0.
    fn purge_step(&self, purge_id: Uuid, budget: usize) -> Result<PurgeProgress>;

    /// Run resumable purge work in one transaction, up to `max_rows` phase operations, checking the caller's
    /// work deadline between small chunks. At least one chunk runs so transaction setup cannot prevent progress.
    /// Empty phase transitions consume budget too. Existing per-row descendant/frame fanout is unchanged.
    ///
    /// The callback must be cheap and must not call the catalog. It supplies cooperative yielding, not a commit
    /// latency guarantee: transaction commit/fsync and final report construction occur after the work window.
    /// Implementations without transaction batching may return after one bounded ordinary step.
    ///
    /// # Errors
    /// What [`Catalog::purge_step`] returns. A callback panic aborts the current transaction in durable catalogs.
    fn purge_batch(
        &self,
        purge_id: Uuid,
        max_rows: usize,
        should_yield: &mut dyn FnMut() -> bool,
    ) -> Result<PurgeProgress> {
        let _ = should_yield;
        self.purge_step(purge_id, max_rows.min(PURGE_BATCH_ROWS))
    }

    /// Starts a purge and runs every step at once, for tests and offline tools. The broker steps purges itself.
    ///
    /// # Errors
    /// What [`Catalog::start_purge`] and [`Catalog::purge_step`] return.
    fn purge(
        &self,
        selector: PurgeSelector,
        reason: &str,
        detail: &str,
        purge_id: Uuid,
        now_ns: u64,
    ) -> Result<PurgeReport> {
        self.start_purge(selector, reason, detail, purge_id, now_ns)?;
        loop {
            if let PurgeProgress::Removed { report, .. } = self.purge_step(purge_id, usize::MAX)? {
                return Ok(report);
            }
        }
    }

    /// A purge's record.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn purge_record(&self, purge_id: Uuid) -> Result<Option<PurgeRecord>>;

    /// Purges in the order they started, after the purge `after`, at most `limit`, as summaries with counts; a
    /// purge's whole report is [`Catalog::purge_record`]'s.
    ///
    /// # Errors
    /// [`CatalogError::Invalid`] for a `limit` of 0, [`CatalogError::NotFound`] when `after` isn't a purge.
    fn purges(&self, after: Option<Uuid>, limit: usize) -> Result<Vec<PurgeSummary>>;

    /// Purges still running, or whose notices are still outstanding, oldest first, for the broker to step or send
    /// again after a restart.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn pending_purges(&self) -> Result<Vec<Uuid>>;

    /// Marks a purge complete: every worker acknowledged, and all shared-copy forgotten bytes/receipts are cleared.
    ///
    /// # Errors
    /// [`CatalogError::NotFound`] when there is no such purge, [`CatalogError::Invalid`] while its steps are still
    /// running; [`CatalogError::Conflict`] while global shared-copy purge cleanup remains.
    fn complete_purge(&self, purge_id: Uuid, now_ns: u64) -> Result<()>;

    /// The workers that keep their own copies, by name, sorted: every name that registered with `keeps_copies`,
    /// until [`Catalog::forget_keeper`] drops it. A purge waits for each of them, connected or not.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn keepers(&self) -> Result<Vec<String>>;

    /// Remembers a keeper's name. A repeat changes nothing.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn add_keeper(&self, name: &str) -> Result<()>;

    /// Drops a keeper's name (the worker was retired or renamed). True when it was known.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn forget_keeper(&self, name: &str) -> Result<bool>;

    /// Records that `keeper` acknowledged a purge's notice. A repeat changes nothing.
    ///
    /// # Errors
    /// [`CatalogError::NotFound`] when there is no such purge, [`CatalogError::Storage`] otherwise.
    fn purge_acked(&self, purge_id: Uuid, keeper: &str) -> Result<()>;

    /// Asks for a scrub even though no purge finished since the last one.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn request_scrub(&self) -> Result<()>;

    /// Whether a scrub is due: a purge finished since the last scrub, or one was requested. A scrub only starts
    /// then; the broker's scrubber (`continuum-broker`'s `scrub` module) decides when to ask for one and how often to
    /// act on a due one.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn scrub_needed(&self) -> Result<bool>;

    /// Runs one step of a scrub, which leaves no purged bytes on disk. A store that keeps freed pages (redb does)
    /// copies its live rows into a fresh file, at most about `budget` rows per step, and the writes made meanwhile
    /// are carried over. The broker runs these steps at low priority with small budgets, so they don't load the
    /// disk. Only the last step, the swap to the fresh file, briefly excludes every other call; it takes about as
    /// long as closing and renaming a file, whatever the catalog's size. With nothing due it returns
    /// [`ScrubProgress::Idle`] and does nothing.
    ///
    /// # Errors
    /// [`CatalogError::Invalid`] for a `budget` of 0, [`CatalogError::Storage`] when the copy or swap fails (the old
    /// file stays in use).
    fn scrub_step(&self, budget: usize) -> Result<ScrubProgress>;

    /// Throws away a running scrub: its fresh file is deleted and the live file stays as it is. A later
    /// [`Catalog::scrub_step`] starts over while one is still due. True when a scrub was running.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] when the fresh file can't be deleted (opening the catalog removes it next time).
    fn abort_scrub(&self) -> Result<bool> {
        Ok(false)
    }

    /// Closes a durable catalog's file cleanly, for a broker that is stopping: a running scrub is thrown away, the
    /// writes in flight finish, and the store is closed so that the next open finds it shut down rather than crashed
    /// (redb otherwise repairs the whole file at the next open, #1383). Every call after it fails. Waits at most
    /// `wait` for reads still holding the file; past that the file is left as a crash would leave it. A catalog with
    /// no file has nothing to do.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] when reads outlast `wait` (the catalog then stays open) or the scrub's fresh file
    /// can't be deleted.
    fn close(&self, wait: std::time::Duration) -> Result<()> {
        let _ = wait;
        Ok(())
    }

    /// The file a durable catalog lives in, its size, and its live bytes, for scheduling scrubs and the broker's
    /// health line. `None` for a catalog with no file (in memory).
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn storage(&self) -> Result<Option<CatalogStorage>> {
        Ok(None)
    }

    /// One step of a sweep that drops the `FRAMES` rows ([`Catalog::commit_chunk`]) nothing needs any more: an
    /// absorbed frame whose chunk ended before `before_ns`, with no children and no subjects, and that no running
    /// purge holds. Such a row only let a late child name the frame as its parent, and the window for that has passed:
    /// a child naming it afterwards is refused as an unknown parent. A frame is still found by time, through its
    /// chunk's frame index (a range purge, a hydrate, whose copy names the chunk), and a frame with children or
    /// subjects keeps its row, so lineage and purges by entity are unchanged. Without its row, a frame's id no longer
    /// resolves ([`Catalog::get`]), and a purge by ids doesn't find it.
    ///
    /// It looks at most `budget` absorbed frames, in (chunk id, frame id) order after `after` (`None` to start), in one
    /// write.
    ///
    /// # Errors
    /// [`CatalogError::Invalid`] for a `budget` of 0, [`CatalogError::Storage`] otherwise.
    fn prune_frames(&self, before_ns: u64, after: Option<(Uuid, Uuid)>, budget: usize) -> Result<PruneProgress>;

    /// Makes every commit so far durable. A durable catalog may leave ordinary commits without an fsync for a moment
    /// (each still atomic and consistent); the broker's writer calls this when it goes idle. Others need not.
    ///
    /// # Errors
    /// Catalog storage failure.
    fn flush_durable(&self) -> Result<()> {
        Ok(())
    }

    /// [`Catalog::prune_frames`] as a cooperative work window (#1803): it looks at up to `max_rows` frames in one
    /// write, but stops early once `should_stop` says so (checked after each frame), and its `next` picks up from
    /// the last frame it looked at. A fixed commit cost then covers as many frames as the window allows, rather than
    /// a row budget that commit latency drives down to one. The callback must be cheap and must not call the catalog.
    /// Implementations without transaction batching may ignore it and take one ordinary step.
    ///
    /// # Errors
    /// What [`Catalog::prune_frames`] returns.
    fn prune_frames_until(
        &self,
        before_ns: u64,
        after: Option<(Uuid, Uuid)>,
        max_rows: usize,
        should_stop: &mut dyn FnMut() -> bool,
    ) -> Result<PruneProgress> {
        let _ = should_stop;
        self.prune_frames(before_ns, after, max_rows)
    }

    /// Saves a durable subscription's position (opaque: the broker owns what it means).
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn set_cursor(&self, name: &str, position: &[u8]) -> Result<()>;

    /// A durable subscription's saved position.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn cursor(&self, name: &str) -> Result<Option<Vec<u8>>>;

    /// Records a monotonic to UTC reading on `boot`.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn record_clock(&self, boot: BootId, point: ClockPoint) -> Result<()>;

    /// The recorded readings around `mono_ns` on `boot`.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn clock_near(&self, boot: BootId, mono_ns: u64) -> Result<ClockNear>;

    /// Saves the scheduler's cost models (opaque bytes the scheduler owns).
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn save_cost_models(&self, bytes: &[u8]) -> Result<()>;

    /// The saved cost models.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn load_cost_models(&self) -> Result<Option<Vec<u8>>>;

    /// Records that a person was forgotten.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn forget(&self, name: ForgottenHash) -> Result<()>;

    /// Stores a durable worker's checkpoint in place of its last one (one row per owner and contract, #1547), only
    /// if every input stream's invalidation revision still equals the one `state` was read under. A purge that
    /// advances one of them later deletes the row in its own transaction.
    ///
    /// # Errors
    /// [`CatalogError::Conflict`] when an input stream isn't registered or its revision has moved (or is at its
    /// ceiling); [`CatalogError::Invalid`] for a state out of bounds (see [`WorkerState`]); [`CatalogError::Storage`].
    fn save_worker_state(&self, state: &WorkerState) -> Result<()>;

    /// A worker's checkpoint, while every input stream's revision still equals the stored one; None otherwise, or
    /// when it has none.
    ///
    /// # Errors
    /// [`CatalogError::Invalid`] for an owner or contract out of bounds; [`CatalogError::Storage`].
    fn load_worker_state(&self, owner: &str, contract: &str) -> Result<Option<WorkerState>>;

    /// Deletes a worker's checkpoint; whether it had one.
    ///
    /// # Errors
    /// [`CatalogError::Invalid`] for an owner or contract out of bounds; [`CatalogError::Storage`].
    fn forget_worker_state(&self, owner: &str, contract: &str) -> Result<bool>;

    /// A bounded, exclusive lexicographic page of opaque forgotten hashes. A page is one read snapshot;
    /// concurrent additions below its cursor are only visible to a new enumeration.
    ///
    /// # Errors
    /// [`CatalogError::Invalid`] unless limit is 1..=1024; [`CatalogError::Storage`].
    fn list_forgotten(&self, after: Option<ForgottenHash>, limit: usize) -> Result<ForgottenPage>;

    /// Whether a person was forgotten.
    ///
    /// # Errors
    /// [`CatalogError::Storage`] only.
    fn is_forgotten(&self, name: ForgottenHash) -> Result<bool>;
}
