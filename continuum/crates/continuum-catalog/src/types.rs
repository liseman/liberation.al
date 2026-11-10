//! What goes in and out of the [`Catalog`](crate::Catalog).

use std::fmt;

use continuum_core::{
    BootId, DomainLayer, HeaderBuf, Interval, PayloadFormat, StageId, StorageLocator, StorageTier, StreamId, Uuid,
};

/// Whether a stream comes from a device or from a worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StreamKind {
    /// Captured from a device by a capture worker.
    Sensor,
    /// Produced by a worker from other payloads.
    Derived,
}

/// One monotonic and one UTC reading taken together, on one boot (decision 9).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClockAnchor {
    /// The boot the monotonic reading belongs to.
    pub boot: BootId,
    /// The host monotonic clock, nanoseconds.
    pub mono_ns: u64,
    /// UTC nanoseconds since the Unix epoch.
    pub utc_ns: u64,
}

/// What a stream is. Registering the same name again with identical info returns the same id.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamInfo {
    /// Unique, like "cam0.rgb" or "face.embedding".
    pub name: String,
    /// Sensor or derived.
    pub kind: StreamKind,
    /// Its payloads' domain layer.
    pub layer: DomainLayer,
    /// How its payloads' bytes are encoded.
    pub format: PayloadFormat,
    /// A raw format like "NV12", or a named shape.
    pub shape_name: String,
    /// The shape's version.
    pub shape_version: u32,
    /// Nominal payloads per second, for the planner's item estimates; `None` when it has no steady rate.
    pub rate_hz: Option<f64>,
    /// The capture worker's clock anchor, for sensor streams.
    pub anchor: Option<ClockAnchor>,
}

/// One frame inside a chunk: when it starts, where its bytes are in the chunk's file, and whether a purge cut it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkFrame {
    /// The frame's `start_ns`.
    pub start_ns: u64,
    /// Its first byte in the chunk.
    pub offset: u64,
    /// Its length in bytes.
    pub length: u64,
    /// Cut by a purge: its bytes are gone, or will be once the tier worker rewrites the chunk, so readers skip it.
    /// A cut frame keeps its place, so frame indexes never change.
    pub cut: bool,
}

impl ChunkFrame {
    /// A frame that hasn't been cut.
    #[must_use]
    pub const fn new(start_ns: u64, offset: u64, length: u64) -> Self {
        Self { start_ns, offset, length, cut: false }
    }
}

/// The indexes of a chunk's frames marked cut.
#[must_use]
pub fn cut_indexes(frames: &[ChunkFrame]) -> Vec<u32> {
    frames.iter().enumerate().filter(|(_, f)| f.cut).map(|(i, _)| u32::try_from(i).unwrap_or(u32::MAX)).collect()
}

/// The time each frame of a chunk holds (its slot), in order: from the frame's start to the next frame's start, or
/// to the end of `chunk_span` for the last frame. When that is empty (an instant, or a frame starting where the chunk
/// ends) the slot is the frame's one nanosecond, `[start, start + 1)`; a frame starting at `u64::MAX` takes
/// `[u64::MAX - 1, u64::MAX)`. Both catalogs use this one rule, and so does anything that follows cut frames.
#[must_use]
pub fn frame_slots(chunk_span: Interval, frames: &[ChunkFrame]) -> Vec<Interval> {
    frames
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let end = frames.get(i + 1).map_or(chunk_span.end_ns(), |next| next.start_ns);
            Interval::new(f.start_ns, end)
                .or_else(|| Interval::covering(f.start_ns, f.start_ns))
                .or_else(|| Interval::new(u64::MAX - 1, u64::MAX))
                .unwrap_or_else(|| unreachable!("[u64::MAX - 1, u64::MAX) is not empty"))
        })
        .collect()
}

/// A payload as the catalog holds it: its header, where its bytes are now (one locator per tier, coldest last), and
/// for a chunk, its frame index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Indexed {
    /// The header.
    pub header: HeaderBuf,
    /// Where its bytes are, in tier order (Tier 0 first). Empty while no copy is placed.
    pub locations: Vec<StorageLocator>,
    /// For a chunk: its frames, in order. `None` for any other payload.
    pub frames: Option<Vec<ChunkFrame>>,
    /// Its commit sequence: where it falls among every payload ever committed to this catalog (see
    /// [`Catalog::insert`](crate::Catalog::insert)). Not the header's `seq`, which orders payloads with the same start.
    pub commit_seq: u64,
}

/// What a [`Catalog::query`](crate::Catalog::query) keeps, all of it ANDed; `None` keeps everything.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct QueryFilter {
    /// Payloads about any of these.
    pub subjects: Option<Vec<Uuid>>,
    /// Payloads starting before this, UTC nanoseconds.
    pub start_before: Option<u64>,
    /// These payloads.
    pub ids: Option<Vec<Uuid>>,
    /// Payloads whose span ([`Interval::covering`]) overlaps this window, as [`Catalog::range`](crate::Catalog::range)
    /// finds them. Set, the query reads the window from the time index (with the stream's longest span as run-up)
    /// instead of scanning historical commits. redb retains a bounded, snapshot-safe window index across pages;
    /// oversized windows use a forward sequence scan from a safe lower bound. The scan cap does not apply.
    pub window: Option<Interval>,
}

/// One page of a [`Catalog::query`](crate::Catalog::query).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeqPage {
    /// Matching payloads, by commit sequence.
    pub items: Vec<Indexed>,
    /// Where to continue (pass it as `after_seq`) when the query stopped at its limit or its scan cap; `None` when
    /// the stream has nothing more past what was looked at.
    pub next_after: Option<u64>,
}

/// A stage finished a span of a source stream (merged into `COVERAGE`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CoverageMark {
    /// The stream the stage read.
    pub source: StreamId,
    /// The stage.
    pub stage: StageId,
    /// What it finished.
    pub span: Interval,
}

/// A payload to index. Its provenance edges and subjects come from its header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewPayload {
    /// The header. Its stream must be registered, its id new, its (stream, start, seq) unused, and its parents
    /// known (indexed, absorbed into a chunk, or earlier in the same batch).
    pub header: HeaderBuf,
    /// Where its bytes are, when they are placed already.
    pub location: Option<StorageLocator>,
    /// The coverage its commit finishes, if any.
    pub coverage: Option<CoverageMark>,
}

/// A chunk that absorbs Tier 0 frames (see [`Catalog::commit_chunk`](crate::Catalog::commit_chunk)).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewChunk {
    /// The chunk's own payload: its header (on the frames' stream, spanning them, with no parents: the frames are
    /// linked through `FRAMES`, not as parents) and its Tier 1 or 2 location.
    pub payload: NewPayload,
    /// The frame index, one entry per absorbed frame, in the same order.
    pub frames: Vec<ChunkFrame>,
}

/// What an id resolves to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Resolved {
    /// A payload with its own header.
    Payload(Indexed),
    /// A raw frame absorbed into a chunk, kept because something was derived from it or it has subjects: its bytes
    /// are `chunk.frames[frame_index]`.
    AbsorbedFrame {
        /// The chunk holding it.
        chunk: Indexed,
        /// Its index in the chunk's frames.
        frame_index: u32,
    },
}

/// Where a range scan stopped: pass it back as `after` for the next page.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RangeCursor {
    /// The last returned payload's `start_ns`.
    pub start_ns: u64,
    /// Its `seq`.
    pub seq: u32,
}

/// One page of a range scan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Page {
    /// Payloads overlapping the window, by (`start_ns`, `seq`).
    pub items: Vec<Indexed>,
    /// Where to continue, when more remain.
    pub next: Option<RangeCursor>,
}

/// One monotonic and UTC reading from `CLOCK_MAP`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClockPoint {
    /// Host monotonic nanoseconds.
    pub mono_ns: u64,
    /// UTC nanoseconds.
    pub utc_ns: u64,
}

/// The `CLOCK_MAP` points around a monotonic time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ClockNear {
    /// The latest point at or before it.
    pub at_or_before: Option<ClockPoint>,
    /// The earliest point after it.
    pub after: Option<ClockPoint>,
}

/// A forgotten person's name, hashed by the caller; the catalog never sees the name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ForgottenHash(pub [u8; 32]);

/// A bounded read-only page of opaque forgotten-name hashes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForgottenPage {
    /// Strictly increasing hashes after the request cursor.
    pub hashes: Vec<ForgottenHash>,
    /// Last returned hash when more remain, otherwise None.
    pub next: Option<ForgottenHash>,
}

/// Maximum forgotten hashes returned by one request.
pub const MAX_FORGOTTEN_PAGE: usize = 1024;

pub(crate) fn forgotten_page(mut hashes: Vec<ForgottenHash>, limit: usize) -> ForgottenPage {
    let more = hashes.len() > limit;
    hashes.truncate(limit);
    let next = more.then(|| hashes.last().copied()).flatten();
    ForgottenPage { hashes, next }
}

/// What a purge removes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PurgeSelector {
    /// Payloads overlapping `span`, on `streams` (every stream when `None`), and everything derived from them.
    Range {
        /// The time to purge.
        span: Interval,
        /// Which streams; `None` for all.
        streams: Option<Vec<StreamId>>,
    },
    /// Payloads with this subject, and everything derived from them.
    Entity(Uuid),
    /// These payloads (or absorbed frames), and everything derived from them.
    Ids(Vec<Uuid>),
    /// These payloads only, aged out by a retention policy: nothing derived from them goes (docs/CONTINUUM-BROKER.md,
    /// "Forgetting"). An id that isn't a payload (an absorbed frame, or unknown), or whose commit sequence is above
    /// `committed_by_seq`, is skipped, so a payload committed after the policy's cutoff (a backdated ingest or an
    /// import, whose start is years old) is never aged out early.
    AgeOut {
        /// The payloads to age out.
        ids: Vec<Uuid>,
        /// The highest commit sequence that may be aged out ([`Catalog::commit_seq_at`](crate::Catalog::commit_seq_at)
        /// of the policy's cutoff).
        committed_by_seq: u64,
        /// Aged-out spans on one stream at most this far apart are recorded as one (0: each payload's own span), so
        /// aging a per-frame stream leaves a run, not an entry a frame. Time inside a run that held nothing reads as
        /// aged; the exact items are in the report until every keeper acknowledges it.
        merge_gap_ns: u64,
    },
}

impl PurgeSelector {
    /// Whether this is an age-out, which removes only what it names.
    #[must_use]
    pub fn is_age_out(&self) -> bool {
        matches!(self, Self::AgeOut { .. })
    }
}

/// A payload a purge removed, with where its bytes were, so the tier and vector workers can delete them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemovedPayload {
    /// Its id.
    pub id: Uuid,
    /// Its stream.
    pub stream: StreamId,
    /// Where its bytes were, in tier order.
    pub locations: Vec<StorageLocator>,
}

/// A chunk that stays, with frames a purge removed from inside it: the tier worker rewrites the chunk without them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartialChunk {
    /// The chunk's id.
    pub chunk: Uuid,
    /// Its stream.
    pub stream: StreamId,
    /// The frames to cut, by index, in order.
    pub frames: Vec<u32>,
    /// Where the chunk is.
    pub locations: Vec<StorageLocator>,
}

/// What a purge did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PurgeReport {
    /// The purge's record id in `PURGES`.
    pub purge_id: Uuid,
    /// Payloads removed, by id.
    pub removed: Vec<RemovedPayload>,
    /// Absorbed frames removed (their `FRAMES` rows), by id.
    pub removed_frames: Vec<Uuid>,
    /// Chunks that stay but lose frames.
    pub partial_chunks: Vec<PartialChunk>,
}

/// What kind of purge a record is of. An entity purge doesn't keep the entity's id, so the record of forgetting
/// someone doesn't keep a way to find them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PurgeKind {
    /// A range purge, as asked.
    Range {
        /// The time purged.
        span: Interval,
        /// Which streams; `None` for all.
        streams: Option<Vec<StreamId>>,
    },
    /// A purge by entity.
    Entity,
    /// A purge by id: how many ids were asked for (the ids aren't kept).
    Ids {
        /// How many.
        count: usize,
    },
    /// An age-out: how many ids were asked for (the ids aren't kept).
    AgeOut {
        /// How many.
        count: usize,
    },
}

/// One coherent catalog snapshot of a visible chunk and its current copies
/// ([`Catalog::chunk_snapshot`](crate::Catalog::chunk_snapshot)).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkSnapshot {
    /// Header, frame index, commit sequence, and locators from this snapshot.
    pub indexed: Indexed,
    /// Per-frame capture provenance from the same snapshot; None remains unknown.
    pub provenance: Option<crate::ChunkProvenance>,
}

/// One purge in a listing ([`Catalog::purges`](crate::Catalog::purges)): its record without the report, and how
/// much it removed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PurgeSummary {
    /// The purge's id.
    pub purge_id: Uuid,
    /// When it started, UTC nanoseconds.
    pub at_ns: u64,
    /// What kind of purge.
    pub kind: PurgeKind,
    /// Why.
    pub reason: String,
    /// The caller's JSON detail.
    pub detail: String,
    /// Payloads removed so far (all of them, once `finished`).
    pub removed: u64,
    /// Whether its steps have finished.
    pub finished: bool,
    /// When every copy was gone, once acknowledged.
    pub completed_ns: Option<u64>,
}

/// A locator as a purge report keeps it: an inline copy loses its bytes (the report is kept for good, and the bytes
/// were what was forgotten), so it says only that the payload was inline.
#[must_use]
pub fn reported(locator: StorageLocator) -> StorageLocator {
    match locator {
        StorageLocator::Inline { .. } => StorageLocator::Inline { bytes: Vec::new() },
        other => other,
    }
}

/// The `PURGES` record of one purge, from [`Catalog::start_purge`](crate::Catalog::start_purge) on. It keeps the
/// whole report once the rows are gone, so the broker can send the `PurgeNotice`s again after a restart until every
/// worker has acknowledged, then mark it complete.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PurgeRecord {
    /// When it started, UTC nanoseconds.
    pub at_ns: u64,
    /// What kind of purge.
    pub kind: PurgeKind,
    /// Why.
    pub reason: String,
    /// The caller's JSON detail, kept as given. It is kept for good (a scrub copies it), so it must not name a
    /// forgotten person: no subject id, as [`PurgeKind::Entity`] keeps none.
    pub detail: String,
    /// What it removed, and where the bytes were; `None` while its steps are still running.
    pub report: Option<PurgeReport>,
    /// When every copy was gone ([`Catalog::complete_purge`](crate::Catalog::complete_purge)); `None` while
    /// notices are outstanding.
    pub completed_ns: Option<u64>,
    /// The keepers (worker names) that acknowledged its notice ([`Catalog::purge_acked`](crate::Catalog::purge_acked)),
    /// so a broker restarting mid-wait tells only the rest.
    pub acked: Vec<String>,
}

/// Why a catalog call failed.
#[derive(Debug)]
#[non_exhaustive]
pub enum CatalogError {
    /// The stream or payload it names isn't there.
    NotFound(String),
    /// It clashes with what's there: a payload id or (stream, start, seq) already used, or a stream name registered
    /// with different info.
    Conflict(String),
    /// It isn't valid on its own terms: an unknown parent, a chunk whose frame index doesn't match its frames, a
    /// location for the wrong tier.
    Invalid(String),
    /// The store underneath failed.
    Storage(Box<dyn std::error::Error + Send + Sync>),
}

impl fmt::Display for CatalogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(what) => write!(f, "not found: {what}"),
            Self::Conflict(what) => write!(f, "conflict: {what}"),
            Self::Invalid(what) => write!(f, "invalid: {what}"),
            Self::Storage(e) => write!(f, "storage: {e}"),
        }
    }
}

impl std::error::Error for CatalogError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Storage(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}

/// A catalog result.
pub type Result<T> = std::result::Result<T, CatalogError>;

/// Where a running purge is. It can go back to `Collecting` when an insert lands under what the purge is hiding,
/// since that has to be followed too.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PurgePhase {
    /// Finding what to remove: the roots, then everything derived from them. What it finds is hidden from reads at
    /// once.
    Collecting,
    /// Removing rows.
    Removing,
    /// Taking the purged time out of coverage, marking cut frames, and writing the report.
    Finishing,
}

/// A payload or absorbed frame a purge step hid, for the broker's availability index to drop at once.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Doomed {
    /// Its id.
    pub id: Uuid,
    /// Its stream (for an absorbed frame, its chunk's).
    pub stream: StreamId,
    /// The time it held (for an absorbed frame, its slot in the chunk).
    pub span: Interval,
    /// The tiers its bytes were in (none for an absorbed frame: its bytes are the chunk's).
    pub tiers: Vec<StorageTier>,
    /// It was hidden as it landed (its insert said so with [`Inserted::hidden`]), so a broker that knew that never
    /// counted it as available; this entry only confirms it.
    pub late: bool,
}

/// What [`Catalog::insert`](crate::Catalog::insert) did with one payload.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Inserted {
    /// Its commit sequence.
    pub seq: u64,
    /// It is hidden now: it landed under a running purge (or was hidden by one already, for a repeat), so it isn't
    /// available, and that purge's delta will name it with `late` set.
    pub hidden: bool,
    /// It was indexed already with identical header bytes, so nothing was written: `seq` is its existing sequence.
    /// Or it is a frame a chunk absorbed since, sent again on the chunk's stream at the start the chunk's frame index
    /// holds (the frame's own header is gone, so that is what can be compared): `seq` is then the chunk's.
    pub duplicate: bool,
    /// For a chunk: the indexes of the frames it landed without that no purge will report, a finished purge having
    /// removed the frame or the worker having cut it; no copy covers their slots. A running purge's cuts (a frame it
    /// has hidden, a slot its window reaches) come in its deltas instead. Empty for a payload.
    pub cut: Vec<u32>,
}

/// A frame a purge step cut from a chunk that stays: that slot of the chunk's stream is no longer readable from the
/// chunk's tiers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CutFrame {
    /// The chunk.
    pub chunk: Uuid,
    /// The frame's index in it.
    pub index: u32,
    /// The chunk's stream.
    pub stream: StreamId,
    /// The frame's slot.
    pub slot: Interval,
    /// The tiers the chunk is in.
    pub tiers: Vec<StorageTier>,
}

/// What one purge step changed, so the broker's availability index can follow in the same step without rereading.
/// Each payload or frame is named in exactly one delta, by the purge that hid it first, even when purges overlap.
/// A step's own finds are bounded by its budget; payloads hidden by inserts since the last step come on top.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PurgeDelta {
    /// Hidden by this step.
    pub doomed: Vec<Doomed>,
    /// Frames cut from chunks that stay.
    pub cut_frames: Vec<CutFrame>,
    /// Coverage taken out, by (source stream, stage). A span may be larger than what was covered.
    pub uncovered: Vec<(StreamId, StageId, Interval)>,
}

/// What one [`Catalog::purge_step`](crate::Catalog::purge_step) did and left.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PurgeProgress {
    /// More steps to go.
    Running {
        /// The phase the next step continues.
        phase: PurgePhase,
        /// Rows handled so far.
        done: u64,
        /// What this step changed.
        delta: PurgeDelta,
    },
    /// Every row is gone: the report is in the record, and the notices can go out.
    Removed {
        /// What the purge removed, and where the bytes were.
        report: PurgeReport,
        /// What this last step changed (empty when asked again after the purge finished).
        delta: PurgeDelta,
    },
}

/// What one [`Catalog::prune_frames`](crate::Catalog::prune_frames) step did.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PruneProgress {
    /// Absorbed frames looked at.
    pub scanned: usize,
    /// Their `FRAMES` rows dropped.
    pub pruned: usize,
    /// Where the next step starts (the last (chunk, frame) looked at), or `None` once the sweep reached the end.
    pub next: Option<(Uuid, Uuid)>,
}

/// What one [`Catalog::scrub_step`](crate::Catalog::scrub_step) left.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScrubProgress {
    /// Nothing to scrub: no purge has finished since the last scrub, and none was requested.
    Idle,
    /// Copying the live rows into a fresh file.
    Copying {
        /// Rows copied so far.
        copied: u64,
    },
    /// The fresh file replaced the old one; no purged bytes are left on disk.
    Swapped,
}

/// How much disk a file-backed catalog takes and how much of it is live rows, from
/// [`Catalog::storage`](crate::Catalog::storage). Cheap to read: a file `stat` and one `META` get.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CatalogStorage {
    /// The catalog file.
    pub path: std::path::PathBuf,
    /// The file's length. redb grows it in steps and only a scrub shrinks it.
    pub file_bytes: u64,
    /// The blocks the file holds on disk (less than `file_bytes` where redb's growth steps are still sparse).
    pub disk_bytes: u64,
    /// The bytes of every live key and value outside `META`, kept exact in each write transaction, or `None` until
    /// the file's first scrub counts them.
    pub live_bytes: Option<u64>,
    /// The length so far of a running scrub's fresh file.
    pub scrub_file_bytes: Option<u64>,
    /// The last scrub that swapped in this file's history, if any.
    pub last_scrub: Option<LastScrub>,
    /// The last scrub that swapped with at least [`LastScrub::DENSITY_MIN_LIVE`] live bytes: the one whose fresh file
    /// says how densely a copy packs. A smaller scrub's file is mostly fixed overhead, so its density overstates a
    /// large catalog's and hides its waste.
    pub density_scrub: Option<LastScrub>,
}

/// What the last swapped scrub did, kept in `META` so it outlives restarts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LastScrub {
    /// When the swap committed, in UTC nanoseconds.
    pub at_ns: u64,
    /// The old file's length.
    pub file_before: u64,
    /// The fresh file's length right after the swap.
    pub file_after: u64,
    /// The live bytes the fresh file held at the swap.
    pub live_bytes: u64,
    /// Rows the copy and the carry-over wrote.
    pub rows: u64,
}

impl LastScrub {
    const LEN: usize = 40;
    /// The live bytes a scrub needs for its density to stand for the catalog's: 1 GiB.
    pub const DENSITY_MIN_LIVE: u64 = 1 << 30;

    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(Self::LEN);
        for v in [self.at_ns, self.file_before, self.file_after, self.live_bytes, self.rows] {
            b.extend_from_slice(&v.to_be_bytes());
        }
        b
    }

    pub(crate) fn decode(b: &[u8]) -> Option<Self> {
        if b.len() != Self::LEN {
            return None;
        }
        let at = |i: usize| b.get(i * 8..i * 8 + 8).and_then(|s| s.try_into().ok()).map(u64::from_be_bytes);
        Some(Self { at_ns: at(0)?, file_before: at(1)?, file_after: at(2)?, live_bytes: at(3)?, rows: at(4)? })
    }
}

/// One table's share of the catalog file (redb's accounting), from [`Catalog::storage_stats`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TableStorage {
    /// The table.
    pub name: String,
    /// Its entries.
    pub rows: u64,
    /// The B-tree's height.
    pub height: u32,
    /// Leaf pages.
    pub leaf_pages: u64,
    /// Branch pages.
    pub branch_pages: u64,
    /// Key and value bytes.
    pub stored_bytes: u64,
    /// Page headers and per-entry offsets.
    pub metadata_bytes: u64,
    /// Unused space inside its pages.
    pub fragmented_bytes: u64,
}

/// Where a catalog file's bytes go, from one read snapshot.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StorageStats {
    /// The page size.
    pub page_size: u64,
    /// The file's length, including space reserved but never written (sparse on disk).
    pub file_bytes: u64,
    /// The blocks the file holds on disk.
    pub disk_bytes: u64,
    /// Every table, largest first.
    pub tables: Vec<TableStorage>,
}

/// A derived payload's dependency on a stretch of a stream: what it was made from, named by source time instead
/// of one parent edge per frame. A purge that removes anything on `stream` overlapping `span` (a payload, or frames cut
/// from a chunk) takes the dependent too, and everything derived from it; an age-out doesn't. Conservative by design:
/// same-time payloads it didn't use take it as well.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dependency {
    /// The stream depended on.
    pub stream: StreamId,
    /// The stretch of it.
    pub span: Interval,
    /// The stream's invalidation revision the producer read before it checked what it used
    /// ([`Catalog::invalidation_revision`](crate::Catalog::invalidation_revision)). The insert is refused (Conflict) if it
    /// moved since: a purge that ran in between would otherwise be missed, a dependency naming no surviving id.
    pub revision: u64,
}

/// One payload's dependencies as the catalogs store them: sorted, one per stream and start, the longest end kept
/// (it overlaps everything a shorter one with that start does, so the purge cascade is the same), and the stored
/// set a repeat is compared with.
///
/// # Errors
///
/// [`CatalogError::Conflict`](crate::CatalogError::Conflict) when two name one stream at different revisions: one
/// of them is stale.
pub fn normalize_dependencies(deps: &[Dependency]) -> Result<Vec<Dependency>> {
    let mut sorted = deps.to_vec();
    sorted.sort_unstable_by_key(|d| (d.stream.0, d.span.start_ns(), d.span.end_ns()));
    let mut out: Vec<Dependency> = Vec::with_capacity(sorted.len());
    for d in sorted {
        if let Some(other) = out.iter().rev().find(|o| o.stream == d.stream)
            && other.revision != d.revision
        {
            return Err(crate::CatalogError::Conflict(format!(
                "stream {} is named at revisions {} and {}: one was read before a change",
                d.stream.0, other.revision, d.revision
            )));
        }
        match out.last_mut() {
            Some(last) if last.stream == d.stream && last.span.start_ns() == d.span.start_ns() => *last = d,
            _ => out.push(d),
        }
    }
    Ok(out)
}
