//! The redb catalog's tables. Every table maps bytes to bytes, with fixed-width big-endian keys, so byte order is key
//! order and one copy routine serves every table (the scrub copies them all into a fresh file).
//!
//! Multimaps (`PROV_CHILDREN`, `SUBJECTS`, and the work tables) are plain tables whose key is both parts and whose
//! value is empty, so "everything under a parent" is a prefix range.

use redb::TableDefinition;

pub(crate) type Def = TableDefinition<'static, &'static [u8], &'static [u8]>;

macro_rules! tables {
    ($($name:ident = $label:literal;)*) => {
        $(pub(crate) const $name: Def = TableDefinition::new($label);)*
        /// Every table, for the scrub's copy.
        pub(crate) const ALL: &[Def] = &[$($name),*];
    };
}

tables! {
    // stream id (u32) -> StreamInfo
    STREAMS = "streams";
    // name -> stream id (u32)
    STREAM_NAMES = "stream_names";
    // stream id (u32) -> the longest span indexed on it (u64), so a range scan knows how far back to start
    STREAM_LONGEST = "stream_longest";
    // (stream u32, start u64, seq u32) -> payload id
    TIME_INDEX = "time_index";
    // payload id -> PayloadHeader bytes
    PAYLOADS = "payloads";
    // payload id -> its commit sequence (u64)
    PAYLOAD_SEQ = "payload_seq";
    // (stream u32, commit sequence u64) -> payload id, for queries in commit order
    STREAM_SEQ = "stream_seq";
    // (payload id, tier u8) -> StorageLocator bytes
    LOCATIONS = "locations";
    // chunk id -> its frame index
    CHUNKS = "chunks";
    // absorbed frame id -> (chunk id, index u32, subject ids)
    FRAMES = "frames";
    // (chunk id, absorbed frame id) -> (): the frames of a chunk that kept a FRAMES row
    CHUNK_FRAMES = "chunk_frames";
    // child id -> parent ids
    PROV_PARENTS = "prov_parents";
    // (parent id, child id) -> ()
    PROV_CHILDREN = "prov_children";
    // (entity id, payload or frame id) -> ()
    SUBJECTS = "subjects";
    // (source stream u32, stage u32, start u64) -> end u64
    COVERAGE = "coverage";
    // durable subscription name -> position
    CURSORS = "cursors";
    // (boot id, monotonic ns u64) -> UTC ns u64
    CLOCK_MAP = "clock_map";
    // name hash (32 bytes) -> ()
    FORGOTTEN = "forgotten";
    // (owner, contract), length-prefixed -> a durable worker's checkpoint, overwritten in place (worker_state.rs)
    WORKER_STATE = "worker_state";
    // (input stream u32, worker key) -> (): the workers whose state a purge of that stream deletes
    WORKER_STATE_INPUTS = "worker_state_inputs";
    // purge id -> PurgeRecord
    PURGES = "purges";
    // (sequence u64) -> purge id, in start order
    PURGE_ORDER = "purge_order";
    // the catalog's own settings: "format", "recording_era_start_seq", the restart stamp, "cost_models", and so on
    META = "meta";
    // ---- running purges ----
    // purge id -> the job's state
    PURGE_JOBS = "purge_jobs";
    // id -> the purge that hid it (every read checks this)
    DOOMED = "doomed";
    // (purge id, id) -> (): hidden and not yet removed
    PURGE_WORK = "purge_work";
    // (purge id, sequence u64) -> id: hidden and not yet followed to what was derived from it
    PURGE_FRONTIER = "purge_frontier";
    // (chunk id, purge id, frame index u32) -> slot: frames to cut, by chunk, for inserts to check against
    PURGE_CUTS = "purge_cuts";
    // (purge id, chunk id, frame index u32) -> slot: the same, by purge
    PURGE_CUT_ORDER = "purge_cut_order";
    // (purge id, sequence u64) -> Doomed: hidden by an insert, for the next step's delta
    PURGE_LATE = "purge_late";
    // (purge id, sequence u64) -> CutFrame: cut by an insert, for the next step's delta
    PURGE_LATE_CUTS = "purge_late_cuts";
    // (purge id, index u64) -> id: a purge by ids' list, read a budget at a time
    PURGE_IDS = "purge_ids";
    // purge id -> payloads removed so far (u64), for purge listings
    PURGE_COUNTS = "purge_counts";
    // (purge id, id) -> RemovedPayload: the report so far
    PURGE_REMOVED = "purge_removed";
    // (purge id, frame id) -> (): absorbed frames removed so far
    PURGE_REMOVED_FRAMES = "purge_removed_frames";
    // ---- age-out (docs/CONTINUUM.md, "Aging") ----
    // UTC ns u64 -> head commit sequence u64 at that time, about one a minute
    COMMIT_CLOCK = "commit_clock";
    // (stream u32, start u64) -> end u64: spans an age-out removed, merged
    AGED_OUT = "aged_out";
    // ---- frame provenance ----
    // chunk id -> versioned per-frame device sequence, capture epoch, and stamp flags, fixed at commit
    CHUNK_PROVENANCE = "chunk_provenance";
    // Native media source -> certified-history flag, current source generation, longest original hull.
    RECORDING_META = "recording_meta";
    // (stream, first source start, original id) -> bound/provisional/unknown multi-run witness.
    RECORDING_ROWS = "recording_rows";
    // Original id -> recording row key, retained after payload/chunk removal.
    RECORDING_IDS = "recording_ids";
    // (original subject, input id) -> (): survives payload age-out, bounded by witness attribution limit.
    RECORDING_SUBJECTS = "recording_subjects";
    // Range purge -> bounded original-witness sweep cursor and source snapshot cutoff.
    RECORDING_PURGE = "recording_purge";
    // (stream u32, start u64, dependent payload id) -> end u64: a derived payload's dependency on a stretch of a
    // stream (#1535), for a purge to find what depended on what it removes
    DEPS = "deps";
    // (dependent payload id, stream u32, start u64) -> []: its dependencies, to drop them when it goes
    DEPS_OF = "deps_of";
    // stream id (u32) -> the longest dependency span on it (u64), so an overlap scan knows how far back to start
    DEPS_LONGEST = "deps_longest";
    // ---- planned restarts (restart.rs) ----
    // payload id -> (): every payload with a Tier 0 location
    HOT_IDS = "restart_hot_ids";
    // "current" -> the broker's opaque index projection, stamped with the catalog revision it was saved at
    RESTART_PROJECTION = "restart_projection";
}
