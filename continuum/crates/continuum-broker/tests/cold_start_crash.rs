//! Crash-restart parity tests for cold-start availability reconstruction.
//!
//! At random points during ingest, `commit_chunk`, `extend_chunk`, `rewrite_chunk`, stepped purges, and restart
//! projection checkpoints, dropping [`RedbCatalog`] without a clean close and recovering the availability index must
//! produce the exact same [`AvailabilityIndex::snapshot`], stream table, [`LoadStats`], and cleared Tier 0 state as a
//! full reference paged rebuild (`index::load_paged`).

use std::path::Path;

use continuum_broker::index::{self, AvailabilityIndex, LoadStats, clear_stale_tier0};
use continuum_catalog::{
    Catalog, ChunkFrame, CoverageMark, Dependency, MAX_RESTART_PROJECTION_BYTES, NewChunk, NewPayload, PurgeProgress,
    PurgeSelector, RedbCatalog, StreamInfo, StreamKind,
};
use continuum_core::{
    DomainLayer, HeaderBuf, HeaderSpec, Interval, PayloadFormat, StageId, StorageLocator, StorageTier, StreamId, Uuid,
};
use proptest::prelude::*;

fn uid(tag: u8, n: u32) -> Uuid {
    // Realistic UUIDv7 layout: 48-bit monotonic timestamp prefix advancing with `(n, tag)` plus deterministic
    // SplitMix64 low bits so key bisection sees realistic time-ordered UUIDv7 prefixes rather than four byte-0 tags.
    let ts_ms = 0x0193_0000_0000_u64 + (u64::from(n) << 8) + u64::from(tag);
    let mut z = ts_ms.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    let rand = z ^ (z >> 31);
    let rand_bytes = rand.to_be_bytes();

    let mut b = [0u8; 16];
    b[0..6].copy_from_slice(&ts_ms.to_be_bytes()[2..8]);
    b[6] = 0x70 | (rand_bytes[0] & 0x0f);
    b[7] = rand_bytes[1];
    b[8] = 0x80 | (rand_bytes[2] & 0x3f);
    b[9..15].copy_from_slice(&rand_bytes[2..8]);
    b[15] = tag;
    Uuid(b)
}

fn iv(a: u64, b: u64) -> Interval {
    Interval::new(a, b).unwrap()
}

fn shm(slot: u32) -> StorageLocator {
    StorageLocator::Shm {
        arena: "crash-test-arena".into(),
        arena_generation: 1,
        slab_class: 0,
        slot,
        slot_generation: 1,
        offset: 0,
        length: 1,
    }
}

fn ring_loc(seq: u64) -> StorageLocator {
    StorageLocator::Ring { arena: "crash-test-arena".into(), arena_generation: 1, ring: 0, first_seq: seq, count: 1 }
}

fn chunk_loc(tier: StorageTier, label: &str, offset: u64) -> StorageLocator {
    StorageLocator::Chunk { tier, uri: label.into(), offset, length: 64 }
}

/// Runs the exact startup restore-or-rebuild decision from `Server::bind_inner`, followed by `clear_stale_tier0`.
fn startup_recover(
    catalog: &RedbCatalog,
    page: usize,
) -> Result<(AvailabilityIndex, usize, bool), continuum_catalog::CatalogError> {
    let pending = catalog.pending_purges()?;
    let cached =
        if pending.is_empty() { catalog.load_restart_projection(MAX_RESTART_PROJECTION_BYTES)?.ok() } else { None };
    let restored =
        cached.and_then(|c| AvailabilityIndex::restore_restart_projection(&c.bytes).ok().map(|i| (i, c.hot_ids)));
    let (index, hot_ids, used_projection) = match restored {
        Some((index, hot_ids)) => (index, hot_ids, true),
        None => {
            let (index, stats) = index::load(catalog, page)?;
            (index, stats.stale_tier0, false)
        }
    };
    let cleared = clear_stale_tier0(catalog, &hot_ids)?;
    Ok((index, cleared, used_projection))
}

fn assert_same_names(a: &AvailabilityIndex, b: &AvailabilityIndex) {
    assert_eq!(a.streams(), b.streams());
    for s in a.streams() {
        let name = b.name_of(s);
        assert_eq!(a.name_of(s), name, "stream {s:?} name mismatch");
        if let Some(ref n) = name {
            assert_eq!(a.stream_named(n), b.stream_named(n), "lookup for stream {n} mismatch");
        }
    }
}

/// Copies a crashed catalog file (closed uncleanly by dropping its handle) and compares:
/// 1. Fast snapshot-backed `index::load` + `clear_stale_tier0`,
/// 2. Lower-threshold sharded scan (`min_parallel_payloads = 1`) and parallel apply (`min_parallel_copies = 1`),
///    each individually and together,
/// 3. Full broker startup recovery (`startup_recover`, testing checkpoint restore or fallback), and
/// 4. Full paged reference rebuild (`index::load_paged` + `clear_stale_tier0`).
fn assert_crash_parity(crashed_path: &Path, page: usize) {
    let dir = tempfile::tempdir().unwrap();
    let full_path = dir.path().join("full.redb");
    let fast_path = dir.path().join("fast.redb");
    let startup_path = dir.path().join("startup.redb");
    std::fs::copy(crashed_path, &full_path).unwrap();
    std::fs::copy(crashed_path, &fast_path).unwrap();
    std::fs::copy(crashed_path, &startup_path).unwrap();

    let full_cat = RedbCatalog::open(&full_path).unwrap();
    let (full_idx, full_stats): (AvailabilityIndex, LoadStats) = index::load_paged(&full_cat, page).unwrap();
    let full_snap = full_idx.snapshot();
    let full_cleared = clear_stale_tier0(&full_cat, &full_stats.stale_tier0).unwrap();

    let fast_cat = RedbCatalog::open(&fast_path).unwrap();

    // Exercise the sharded parallel scan on small random histories, compared against the paged reference path.
    let (sharded_snap, _) = fast_cat.availability_snapshot_for_test(1).unwrap();
    let (sharded_idx, sharded_stats) = index::load_from_snapshot_for_test(sharded_snap, usize::MAX);
    assert_eq!(sharded_idx.snapshot(), full_snap, "sharded scan diverged from full paged rebuild");
    assert_eq!(sharded_stats, full_stats, "sharded scan LoadStats diverged from full paged rebuild");
    assert_same_names(&sharded_idx, &full_idx);

    // Exercise the parallel per-stream apply on small random histories, compared against the paged reference path.
    let default_snap =
        fast_cat.availability_snapshot().unwrap().expect("RedbCatalog must supply availability_snapshot");
    let (par_apply_idx, par_apply_stats) = index::load_from_snapshot_for_test(default_snap, 1);
    assert_eq!(par_apply_idx.snapshot(), full_snap, "parallel apply diverged from full paged rebuild");
    assert_eq!(par_apply_stats, full_stats, "parallel apply LoadStats diverged from full paged rebuild");
    assert_same_names(&par_apply_idx, &full_idx);

    // Exercise both lowered thresholds together before running production `index::load` + `clear_stale_tier0`.
    let (both_snap, _) = fast_cat.availability_snapshot_for_test(1).unwrap();
    let (both_idx, both_stats) = index::load_from_snapshot_for_test(both_snap, 1);
    assert_eq!(both_idx.snapshot(), full_snap, "combined sharded scan + parallel apply diverged from paged rebuild");
    assert_eq!(both_stats, full_stats, "combined sharded scan + parallel apply LoadStats diverged");
    assert_same_names(&both_idx, &full_idx);

    let (fast_idx, fast_stats) = index::load(&fast_cat, page).unwrap();
    let fast_cleared = clear_stale_tier0(&fast_cat, &fast_stats.stale_tier0).unwrap();

    assert_eq!(fast_idx.snapshot(), full_snap, "fast scan snapshot diverged from full paged rebuild");
    assert_eq!(fast_stats, full_stats, "fast scan LoadStats diverged from full paged rebuild");
    assert_eq!(fast_cleared, full_cleared, "fast scan cleared Tier 0 count diverged from full paged rebuild");
    assert_same_names(&fast_idx, &full_idx);

    let startup_cat = RedbCatalog::open(&startup_path).unwrap();
    let (startup_idx, startup_cleared, _) = startup_recover(&startup_cat, page).unwrap();
    assert_eq!(startup_idx.snapshot(), full_idx.snapshot(), "startup recovery diverged from full paged rebuild");
    assert_eq!(startup_cleared, full_cleared, "startup recovery cleared Tier 0 count diverged");
    assert_same_names(&startup_idx, &full_idx);

    // After clearing obsolete Tier 0 locators, a subsequent reference rebuild must see zero stale Tier 0 rows.
    let (after_idx, after_stats) = index::load_paged(&startup_cat, page).unwrap();
    assert!(after_stats.stale_tier0.is_empty(), "stale Tier 0 rows survived startup cleanup");
    assert_eq!(after_idx.snapshot(), full_idx.snapshot());
}

#[derive(Clone, Copy, Debug)]
enum CheckpointMode {
    Valid,
    InvalidatedBeforePublish,
    CorruptBytes,
}

#[derive(Clone, Debug)]
enum PurgeKind {
    Range { start: u64, len: u64, all_streams: bool },
    Entity(bool),
    Ids(Vec<usize>),
    AgeOut(Vec<usize>),
}

#[derive(Clone, Debug)]
enum Op {
    IngestSensor { rated: bool, count: u8, instant: bool, loc_kind: u8, also_warm: bool, same_time_seq: bool },
    IngestDerived { parent_pick: usize, subject: Option<bool>, stage: Option<u32>, with_dep: bool, loc_kind: u8 },
    CommitChunk { rated: bool, take: u8, cut_mask: u8, collide_first_key: bool, cold: bool },
    ExtendChunk { chunk_pick: usize, take: u8, cut_mask: u8, cold: bool },
    RewriteChunk { chunk_pick: usize, cold: bool },
    ClearLocation { payload_pick: usize, tier: u8 },
    StartPurge(PurgeKind),
    StepPurge { purge_pick: usize, budget: usize, finish: bool },
    Checkpoint(CheckpointMode),
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        3 => (any::<bool>(), 1u8..=5, any::<bool>(), 0u8..6, any::<bool>(), any::<bool>()).prop_map(
            |(rated, count, instant, loc_kind, also_warm, same_time_seq)| Op::IngestSensor {
                rated,
                count,
                instant,
                loc_kind,
                also_warm,
                same_time_seq,
            }
        ),
        2 => (
            0usize..128,
            prop::option::of(any::<bool>()),
            prop::option::of(1u32..=2),
            any::<bool>(),
            0u8..5,
        )
            .prop_map(|(parent_pick, subject, stage, with_dep, loc_kind)| Op::IngestDerived {
                parent_pick,
                subject,
                stage,
                with_dep,
                loc_kind,
            }),
        2 => (any::<bool>(), 1u8..=4, any::<u8>(), any::<bool>(), any::<bool>()).prop_map(
            |(rated, take, cut_mask, collide_first_key, cold)| Op::CommitChunk {
                rated,
                take,
                cut_mask,
                collide_first_key,
                cold,
            }
        ),
        2 => (0usize..32, 1u8..=3, any::<u8>(), any::<bool>()).prop_map(
            |(chunk_pick, take, cut_mask, cold)| Op::ExtendChunk { chunk_pick, take, cut_mask, cold }
        ),
        1 => (0usize..32, any::<bool>()).prop_map(|(chunk_pick, cold)| Op::RewriteChunk { chunk_pick, cold }),
        1 => (0usize..128, 0u8..=2).prop_map(|(payload_pick, tier)| Op::ClearLocation { payload_pick, tier }),
        2 => prop_oneof![
            (0u64..400, 10u64..160, any::<bool>())
                .prop_map(|(start, len, all_streams)| Op::StartPurge(PurgeKind::Range { start, len, all_streams })),
            any::<bool>().prop_map(|which| Op::StartPurge(PurgeKind::Entity(which))),
            proptest::collection::vec(0usize..128, 1..=3).prop_map(|picks| Op::StartPurge(PurgeKind::Ids(picks))),
            proptest::collection::vec(0usize..128, 1..=3).prop_map(|picks| Op::StartPurge(PurgeKind::AgeOut(picks))),
        ],
        3 => (0usize..16, 1usize..=5, any::<bool>())
            .prop_map(|(purge_pick, budget, finish)| Op::StepPurge { purge_pick, budget, finish }),
        2 => prop_oneof![
            Just(Op::Checkpoint(CheckpointMode::Valid)),
            Just(Op::Checkpoint(CheckpointMode::InvalidatedBeforePublish)),
            Just(Op::Checkpoint(CheckpointMode::CorruptBytes)),
        ],
    ]
}

#[derive(Clone)]
struct RawFrame {
    id: Uuid,
    start_ns: u64,
    end_ns: u64,
    seq: u32,
}

#[derive(Clone)]
struct TrackedChunk {
    id: Uuid,
    rated: bool,
    stream: StreamId,
    start_ns: u64,
    end_ns: u64,
    seq: u32,
    frames: Vec<ChunkFrame>,
}

fn sensor_locator(kind: u8, n: u32) -> Option<StorageLocator> {
    match kind % 6 {
        0 => Some(shm(n)),
        1 => Some(ring_loc(u64::from(n))),
        2 => Some(StorageLocator::Inline { bytes: vec![0x42; 32] }),
        3 => Some(chunk_loc(StorageTier::Tier1Warm, "warm-raw", u64::from(n) * 64)),
        4 => Some(chunk_loc(StorageTier::Tier2Cold, "cold-raw", u64::from(n) * 64)),
        _ => None,
    }
}

fn derived_locator(kind: u8, n: u32) -> Option<StorageLocator> {
    match kind % 5 {
        0 => Some(shm(n)),
        1 => Some(StorageLocator::Inline { bytes: vec![0x7a; 24] }),
        2 => Some(chunk_loc(StorageTier::Tier1Warm, "warm-derived", u64::from(n) * 64)),
        3 => Some(chunk_loc(StorageTier::Tier2Cold, "cold-derived", u64::from(n) * 64)),
        _ => None,
    }
}

struct Harness {
    cam_rated: StreamId,
    cam_plain: StreamId,
    derived: StreamId,
    next_id: u32,
    next_purge: u32,
    rated_time: u64,
    plain_time: u64,
    derived_time: u64,
    clock_ns: u64,
    rated_raw: Vec<RawFrame>,
    plain_raw: Vec<RawFrame>,
    chunks: Vec<TrackedChunk>,
    all_ids: Vec<Uuid>,
    purges: Vec<Uuid>,
}

impl Harness {
    fn new(c: &RedbCatalog) -> Self {
        let cam_rated = c
            .register_stream(StreamInfo {
                name: "cam.rated".into(),
                kind: StreamKind::Sensor,
                layer: DomainLayer::Layer1Raw,
                format: PayloadFormat::RawBytes,
                shape_name: "raw".into(),
                shape_version: 1,
                // 10 ns nominal period (`1e9 / 1e8 = 10`) so instants merge and overlap cleanly on small grids.
                rate_hz: Some(1e8),
                anchor: None,
            })
            .unwrap();
        let cam_plain = c
            .register_stream(StreamInfo {
                name: "cam.plain".into(),
                kind: StreamKind::Sensor,
                layer: DomainLayer::Layer1Raw,
                format: PayloadFormat::RawBytes,
                shape_name: "raw".into(),
                shape_version: 1,
                rate_hz: None,
                anchor: None,
            })
            .unwrap();
        let derived = c
            .register_stream(StreamInfo {
                name: "vision.crops".into(),
                kind: StreamKind::Derived,
                layer: DomainLayer::Layer1Raw,
                format: PayloadFormat::RawBytes,
                shape_name: "crop".into(),
                shape_version: 1,
                rate_hz: None,
                anchor: None,
            })
            .unwrap();
        Self {
            cam_rated,
            cam_plain,
            derived,
            next_id: 1,
            next_purge: 1,
            rated_time: 10,
            plain_time: 10,
            derived_time: 10,
            clock_ns: 1_000,
            rated_raw: Vec::new(),
            plain_raw: Vec::new(),
            chunks: Vec::new(),
            all_ids: Vec::new(),
            purges: Vec::new(),
        }
    }

    fn alloc_id(&mut self, tag: u8) -> (u32, Uuid) {
        let n = self.next_id;
        self.next_id += 1;
        (n, uid(tag, n))
    }

    fn apply(&mut self, c: &RedbCatalog, op: Op) {
        match op {
            Op::IngestSensor { rated, count, instant, loc_kind, also_warm, same_time_seq } => {
                let stream = if rated { self.cam_rated } else { self.cam_plain };
                let mut batch = Vec::new();
                let mut staged = Vec::new();
                for i in 0..count {
                    let (n, id) = self.alloc_id(if rated { 0x01 } else { 0x02 });
                    let cursor = if rated { &mut self.rated_time } else { &mut self.plain_time };
                    let (start_ns, seq) = if same_time_seq && i > 0 {
                        (*cursor, u32::from(i))
                    } else {
                        *cursor += 10;
                        (*cursor, 0)
                    };
                    let end_ns = if instant { start_ns } else { start_ns + 10 };
                    let mut spec = HeaderSpec::raw(id, stream, start_ns, end_ns);
                    spec.seq = seq;
                    let header = HeaderBuf::build(&spec).unwrap();
                    let location = sensor_locator(loc_kind.wrapping_add(i), n);
                    batch.push(NewPayload { header, location, coverage: None });
                    staged.push((n, RawFrame { id, start_ns, end_ns, seq }));
                }
                if c.insert(batch).is_ok() {
                    for (n, frame) in staged {
                        if also_warm {
                            let _ = c.set_location(frame.id, chunk_loc(StorageTier::Tier1Warm, "mirror", u64::from(n)));
                        }
                        self.all_ids.push(frame.id);
                        if rated {
                            self.rated_raw.push(frame);
                        } else {
                            self.plain_raw.push(frame);
                        }
                    }
                }
            }
            Op::IngestDerived { parent_pick, subject, stage, with_dep, loc_kind } => {
                let parents = if self.all_ids.is_empty() {
                    Vec::new()
                } else {
                    vec![self.all_ids[parent_pick % self.all_ids.len()]]
                };
                let (n, id) = self.alloc_id(0x03);
                self.derived_time += 7;
                let start_ns = self.derived_time;
                let end_ns = start_ns + 12;
                let mut spec = HeaderSpec::raw(id, self.derived, start_ns, end_ns);
                spec.parents = parents;
                if let Some(which) = subject {
                    spec.subjects = vec![uid(0xee, u32::from(which) + 900)];
                }
                let coverage = stage.map(|st| CoverageMark {
                    source: self.cam_rated,
                    stage: StageId(st),
                    span: iv(start_ns, end_ns),
                });
                let payload = NewPayload {
                    header: HeaderBuf::build(&spec).unwrap(),
                    location: derived_locator(loc_kind, n),
                    coverage,
                };
                let res = if with_dep && let Ok(Some(rev)) = c.invalidation_revision(self.cam_rated) {
                    let dep = Dependency { stream: self.cam_rated, span: iv(start_ns, end_ns), revision: rev };
                    c.insert_dependent(vec![payload], vec![(id, vec![dep])])
                } else {
                    c.insert(vec![payload])
                };
                if res.is_ok() {
                    self.all_ids.push(id);
                }
            }
            Op::CommitChunk { rated, take, cut_mask, collide_first_key, cold } => {
                let pool = if rated { &mut self.rated_raw } else { &mut self.plain_raw };
                let count = usize::from(take).min(pool.len());
                if count == 0 {
                    return;
                }
                let chosen: Vec<RawFrame> = pool.drain(..count).collect();
                let stream = if rated { self.cam_rated } else { self.cam_plain };
                let start_ns = chosen[0].start_ns;
                let last = chosen.last().unwrap();
                let end_ns = last.end_ns.max(last.start_ns + 10);
                let seq = if collide_first_key { chosen[0].seq } else { chosen[0].seq.saturating_add(100) };
                let (n, chunk_id) = self.alloc_id(0x04);
                let mut spec = HeaderSpec::raw(chunk_id, stream, start_ns, end_ns);
                spec.seq = seq;
                let frames: Vec<ChunkFrame> = chosen
                    .iter()
                    .enumerate()
                    .map(|(i, f)| ChunkFrame {
                        start_ns: f.start_ns,
                        offset: (i as u64) * 64,
                        length: 64,
                        cut: (cut_mask & (1 << (i % 8))) != 0,
                    })
                    .collect();
                let absorbed: Vec<Uuid> = chosen.iter().map(|f| f.id).collect();
                let tier = if cold { StorageTier::Tier2Cold } else { StorageTier::Tier1Warm };
                let new_chunk = NewChunk {
                    payload: NewPayload {
                        header: HeaderBuf::build(&spec).unwrap(),
                        location: Some(chunk_loc(tier, "chunk-initial", u64::from(n) * 1024)),
                        coverage: None,
                    },
                    frames: frames.clone(),
                };
                if c.commit_chunk(new_chunk, &absorbed).is_ok()
                    && let Ok(Some(continuum_catalog::Resolved::Payload(indexed))) = c.get(chunk_id)
                    && let Some(actual_frames) = indexed.frames
                {
                    self.all_ids.push(chunk_id);
                    self.chunks.push(TrackedChunk {
                        id: chunk_id,
                        rated,
                        stream,
                        start_ns,
                        end_ns,
                        seq,
                        frames: actual_frames,
                    });
                }
            }
            Op::ExtendChunk { chunk_pick, take, cut_mask, cold } => {
                if self.chunks.is_empty() {
                    return;
                }
                let idx = chunk_pick % self.chunks.len();
                let rated = self.chunks[idx].rated;
                let last_frame_start = self.chunks[idx].frames.last().map_or(self.chunks[idx].start_ns, |f| f.start_ns);
                let pool = if rated { &mut self.rated_raw } else { &mut self.plain_raw };
                let pos = pool.partition_point(|f| f.start_ns < last_frame_start);
                if pos >= pool.len() {
                    return;
                }
                let end_pos = (pos + usize::from(take)).min(pool.len());
                let chosen: Vec<RawFrame> = pool.drain(pos..end_pos).collect();
                let chunk_meta = self.chunks[idx].clone();
                let last = chosen.last().unwrap();
                let new_end = chunk_meta.end_ns.max(last.end_ns).max(last.start_ns + 10);
                let mut spec = HeaderSpec::raw(chunk_meta.id, chunk_meta.stream, chunk_meta.start_ns, new_end);
                spec.seq = chunk_meta.seq;
                let base_len = chunk_meta.frames.len();
                let part_frames: Vec<ChunkFrame> = chosen
                    .iter()
                    .enumerate()
                    .map(|(i, f)| ChunkFrame {
                        start_ns: f.start_ns,
                        offset: ((base_len + i) as u64) * 64,
                        length: 64,
                        cut: (cut_mask & (1 << (i % 8))) != 0,
                    })
                    .collect();
                let absorbed: Vec<Uuid> = chosen.iter().map(|f| f.id).collect();
                let tier = if cold { StorageTier::Tier2Cold } else { StorageTier::Tier1Warm };
                let part = NewChunk {
                    payload: NewPayload {
                        header: HeaderBuf::build(&spec).unwrap(),
                        location: Some(chunk_loc(tier, "chunk-extended", new_end)),
                        coverage: None,
                    },
                    frames: part_frames,
                };
                if let Ok(first_frame) = u32::try_from(base_len)
                    && c.extend_chunk(part, first_frame, &absorbed, None).is_ok()
                    && let Ok(Some(continuum_catalog::Resolved::Payload(indexed))) = c.get(chunk_meta.id)
                    && let Some(actual_frames) = indexed.frames
                {
                    self.chunks[idx].end_ns = new_end;
                    self.chunks[idx].frames = actual_frames;
                }
            }
            Op::RewriteChunk { chunk_pick, cold } => {
                if self.chunks.is_empty() {
                    return;
                }
                let chunk = &self.chunks[chunk_pick % self.chunks.len()];
                if let Ok(Some(continuum_catalog::Resolved::Payload(indexed))) = c.get(chunk.id)
                    && let Some(frames) = indexed.frames
                {
                    let tier = if cold { StorageTier::Tier2Cold } else { StorageTier::Tier1Warm };
                    let _ = c.rewrite_chunk(chunk.id, frames, chunk_loc(tier, "chunk-rewritten", 0));
                }
            }
            Op::ClearLocation { payload_pick, tier } => {
                if self.all_ids.is_empty() {
                    return;
                }
                let id = self.all_ids[payload_pick % self.all_ids.len()];
                let _ = c.clear_location(id, StorageTier(tier));
            }
            Op::StartPurge(kind) => {
                let p_num = self.next_purge;
                self.next_purge += 1;
                let purge_id = uid(0xf0, p_num);
                self.clock_ns += 100;
                let _ = c.mark_commit_clock(self.clock_ns);
                let selector = match kind {
                    PurgeKind::Range { start, len, all_streams } => PurgeSelector::Range {
                        span: iv(start, start + len.max(1)),
                        streams: if all_streams { None } else { Some(vec![self.cam_rated]) },
                    },
                    PurgeKind::Entity(which) => PurgeSelector::Entity(uid(0xee, u32::from(which) + 900)),
                    PurgeKind::Ids(picks) => {
                        if self.all_ids.is_empty() {
                            return;
                        }
                        let ids = picks.into_iter().map(|p| self.all_ids[p % self.all_ids.len()]).collect();
                        PurgeSelector::Ids(ids)
                    }
                    PurgeKind::AgeOut(picks) => {
                        if self.all_ids.is_empty() {
                            return;
                        }
                        let ids = picks.into_iter().map(|p| self.all_ids[p % self.all_ids.len()]).collect();
                        let committed_by_seq = c.head_seq().unwrap_or(0);
                        PurgeSelector::AgeOut { ids, committed_by_seq, merge_gap_ns: 10 }
                    }
                };
                if c.start_purge(selector, "crash-prop", "{}", purge_id, self.clock_ns).is_ok() {
                    self.purges.push(purge_id);
                }
            }
            Op::StepPurge { purge_pick, budget, finish } => {
                if self.purges.is_empty() {
                    return;
                }
                let purge_id = self.purges[purge_pick % self.purges.len()];
                if finish {
                    while matches!(c.purge_step(purge_id, 32), Ok(PurgeProgress::Running { .. })) {}
                } else {
                    let _ = c.purge_step(purge_id, budget.max(1));
                }
            }
            Op::Checkpoint(mode) => {
                let Ok(Some(stamp)) = c.restart_stamp() else { return };
                match mode {
                    CheckpointMode::Valid => {
                        if let Ok((idx, _)) = index::load(c, 64)
                            && let Ok(bytes) = idx.restart_projection_bytes()
                        {
                            let _ = c.save_restart_projection(stamp, &bytes);
                        }
                    }
                    CheckpointMode::InvalidatedBeforePublish => {
                        if let Ok((idx, _)) = index::load(c, 64)
                            && let Ok(bytes) = idx.restart_projection_bytes()
                        {
                            self.clock_ns += 1;
                            let _ = c.mark_commit_clock(self.clock_ns);
                            let published = c.save_restart_projection(stamp, &bytes).unwrap();
                            assert!(!published, "stale restart stamp must never publish a checkpoint");
                        }
                    }
                    CheckpointMode::CorruptBytes => {
                        let _ = c.save_restart_projection(stamp, b"corrupt-restart-projection");
                    }
                }
            }
        }
    }
}

/// Seeds 7,000 payloads (>4,096 payloads and >4,300 durable Tier 1/Tier 2 copies) across sensor and derived streams
/// so production `index::load` exceeds both `PARALLEL_MIN_PAYLOADS` and `PARALLEL_APPLY_MIN_COPIES`.
fn seed_bulk_catalog(catalog: &RedbCatalog, h: &mut Harness) {
    for batch_idx in 0..140u32 {
        let mut batch = Vec::with_capacity(50);
        for k in 0..50u32 {
            let (n, id) = h.alloc_id(u8::try_from((batch_idx + k) % 4).unwrap());
            let stream = match k % 3 {
                0 => h.cam_rated,
                1 => h.cam_plain,
                _ => h.derived,
            };
            let start = u64::from(batch_idx * 500 + k * 10 + 10);
            let end = if k % 3 == 0 { start } else { start + 10 };
            let mut spec = HeaderSpec::raw(id, stream, start, end);
            if k % 11 == 0 {
                spec.subjects = vec![uid(0xee, 900)];
            }
            let pick = (batch_idx + k) % 8;
            let location = match pick {
                0 => Some(shm(n)),
                1 => Some(ring_loc(u64::from(n))),
                2 if (batch_idx + k).is_multiple_of(16) => None,
                2 => Some(StorageLocator::Inline { bytes: vec![0x55; 48] }),
                3..=5 => Some(chunk_loc(StorageTier::Tier1Warm, "warm-bulk", u64::from(n))),
                _ => Some(chunk_loc(StorageTier::Tier2Cold, "cold-bulk", u64::from(n))),
            };
            let coverage = (k % 9 == 0).then(|| CoverageMark {
                source: h.cam_rated,
                stage: StageId((k % 2) + 1),
                span: iv(start, start + 10),
            });
            batch.push(NewPayload { header: HeaderBuf::build(&spec).unwrap(), location, coverage });
            h.all_ids.push(id);
            if stream == h.cam_rated {
                h.rated_raw.push(RawFrame { id, start_ns: start, end_ns: end, seq: 0 });
            } else if stream == h.cam_plain {
                h.plain_raw.push(RawFrame { id, start_ns: start, end_ns: end, seq: 0 });
            }
        }
        catalog.insert(batch).unwrap();
    }
    h.rated_time = 800_000;
    h.plain_time = 800_000;
}

#[test]
fn parallel_sharded_fast_path_without_collision_matches_full_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("parallel_no_collision.redb");
    let catalog = RedbCatalog::open(&path).unwrap();
    let mut h = Harness::new(&catalog);
    seed_bulk_catalog(&catalog, &mut h);

    // Commit, extend, and rewrite chunks WITHOUT colliding on the first frame's `(stream, start_ns, seq)` key so
    // `TIME_INDEX.len() == PAYLOADS.len()` (`present_ids == None`) runs across the 64-shard parallel scan and
    // parallel per-stream `CopySet` apply.
    h.apply(
        &catalog,
        Op::CommitChunk { rated: true, take: 4, cut_mask: 0b0010, collide_first_key: false, cold: false },
    );
    h.apply(&catalog, Op::ExtendChunk { chunk_pick: 0, take: 3, cut_mask: 0b0100, cold: true });
    h.apply(
        &catalog,
        Op::CommitChunk { rated: false, take: 4, cut_mask: 0b0000, collide_first_key: false, cold: false },
    );
    h.apply(&catalog, Op::RewriteChunk { chunk_pick: 1, cold: true });

    // Invalidate a checkpoint and leave range and entity purges mid-step with active `DOOMED` and `CHUNKS` cuts.
    h.apply(&catalog, Op::Checkpoint(CheckpointMode::Valid));
    h.apply(&catalog, Op::StartPurge(PurgeKind::Entity(false)));
    h.apply(&catalog, Op::StepPurge { purge_pick: 0, budget: 3, finish: false });
    h.apply(&catalog, Op::StartPurge(PurgeKind::Range { start: 100, len: 2_500, all_streams: false }));
    h.apply(&catalog, Op::StepPurge { purge_pick: 1, budget: 4, finish: false });

    drop(catalog);

    let check_cat = RedbCatalog::open(&path).unwrap();
    let (snap, used_fallback) = check_cat.availability_snapshot_for_test(4_096).unwrap();
    assert!(!used_fallback, "non-colliding large history must take the present_ids == None fast path");
    assert!(snap.payloads > 6_000, "expected >6,000 live payloads, got {}", snap.payloads);
    drop(check_cat);

    for page in [1, 17, 1024] {
        assert_crash_parity(&path, page);
    }
}

#[test]
fn parallel_sharded_crash_recovery_with_cuts_extend_chunk_and_mid_purge_matches_full_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("parallel_crash.redb");
    let catalog = RedbCatalog::open(&path).unwrap();
    let mut h = Harness::new(&catalog);
    seed_bulk_catalog(&catalog, &mut h);

    // Commit a chunk whose first frame is worker-cut AND shares `(stream, start_ns, seq)` with the chunk,
    // exercising `TIME_INDEX.len() < PAYLOADS.len()`, then extend it with `extend_chunk`.
    h.apply(&catalog, Op::CommitChunk { rated: true, take: 4, cut_mask: 0b0001, collide_first_key: true, cold: false });
    h.apply(&catalog, Op::ExtendChunk { chunk_pick: 0, take: 3, cut_mask: 0b0010, cold: true });

    // Publish a valid checkpoint, then immediately start a range purge and an entity purge and step each once
    // so the crash happens mid-purge with an invalidated checkpoint and active `DOOMED` and `CHUNKS` cuts.
    h.apply(&catalog, Op::Checkpoint(CheckpointMode::Valid));
    h.apply(&catalog, Op::StartPurge(PurgeKind::Entity(false)));
    h.apply(&catalog, Op::StepPurge { purge_pick: 0, budget: 3, finish: false });
    h.apply(&catalog, Op::StartPurge(PurgeKind::Range { start: 100, len: 2_500, all_streams: false }));
    h.apply(&catalog, Op::StepPurge { purge_pick: 1, budget: 4, finish: false });

    // Simulate an unplanned crash by dropping `RedbCatalog` without `close()`.
    drop(catalog);

    let check_cat = RedbCatalog::open(&path).unwrap();
    let (snap, used_fallback) = check_cat.availability_snapshot_for_test(4_096).unwrap();
    assert!(used_fallback, "colliding worker-cut first frame must take the present_ids fallback");
    assert!(snap.payloads > 6_000, "expected >6,000 live payloads, got {}", snap.payloads);
    drop(check_cat);

    for page in [1, 17, 1024] {
        assert_crash_parity(&path, page);
    }
}

#[test]
fn overlapping_rate_expanded_cut_slots_match_full_rebuild_in_both_seq_orders() {
    for chunk_seq_after_copy in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cut_order.redb");
        let catalog = RedbCatalog::open(&path).unwrap();
        let h = Harness::new(&catalog);

        // Two raw frames at the same start_ns (20) on a rated 10 ns period stream, plus one overlapping durable copy
        // either after (`seq = 5`) or before (`seq = 0`, chunk at `seq = 10`) in TIME_INDEX order.
        for seq in [0u32, 1u32] {
            let mut spec = HeaderSpec::raw(uid(0x10, seq + 1), h.cam_rated, 20, 30);
            spec.seq = seq;
            catalog
                .insert(vec![NewPayload {
                    header: HeaderBuf::build(&spec).unwrap(),
                    location: Some(shm(seq + 1)),
                    coverage: None,
                }])
                .unwrap();
        }
        let copy_seq = if chunk_seq_after_copy { 2u32 } else { 20u32 };
        let chunk_seq = if chunk_seq_after_copy { 10u32 } else { 5u32 };

        let mut copy_spec = HeaderSpec::raw(uid(0x11, 1), h.cam_rated, 20, 30);
        copy_spec.seq = copy_seq;
        catalog
            .insert(vec![NewPayload {
                header: HeaderBuf::build(&copy_spec).unwrap(),
                location: Some(chunk_loc(StorageTier::Tier2Cold, "overlap-copy", 0)),
                coverage: None,
            }])
            .unwrap();

        // Both absorbed frames share start_ns = 20 and are cut, producing slots `[20, 21)` and `[20, 30)` which both
        // expand to `[20, 30)` under `rate_hz = 1e8`.
        let mut chunk_spec = HeaderSpec::raw(uid(0x12, 1), h.cam_rated, 20, 30);
        chunk_spec.seq = chunk_seq;
        let frames = vec![
            ChunkFrame { start_ns: 20, offset: 0, length: 0, cut: true },
            ChunkFrame { start_ns: 20, offset: 0, length: 0, cut: true },
        ];
        catalog
            .commit_chunk(
                NewChunk {
                    payload: NewPayload {
                        header: HeaderBuf::build(&chunk_spec).unwrap(),
                        location: Some(chunk_loc(StorageTier::Tier2Cold, "cut-chunk", 0)),
                        coverage: None,
                    },
                    frames,
                },
                &[uid(0x10, 1), uid(0x10, 2)],
            )
            .unwrap();

        drop(catalog);
        assert_crash_parity(&path, 16);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn random_crashes_across_ingest_purge_extend_chunk_and_checkpoint_match_full_rebuild(
        ops in proptest::collection::vec(op_strategy(), 10..36),
        crash_fracs in proptest::collection::vec(0usize..100, 1..=3),
        page in prop_oneof![Just(1usize), Just(7usize), Just(1024usize)],
    ) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prop_crash.redb");
        let mut crash_steps: Vec<usize> = crash_fracs
            .into_iter()
            .map(|f| (f * ops.len()) / 100)
            .collect();
        crash_steps.push(ops.len());
        crash_steps.sort_unstable();
        crash_steps.dedup();

        let mut catalog = Some(RedbCatalog::open(&path).unwrap());
        let mut harness = Harness::new(catalog.as_ref().unwrap());
        let mut next_crash = 0usize;

        for (step, op) in ops.into_iter().enumerate() {
            while next_crash < crash_steps.len() && crash_steps[next_crash] == step {
                // Drop without `Catalog::close` to simulate an unplanned stop at this exact point.
                drop(catalog.take());
                assert_crash_parity(&path, page);
                catalog = Some(RedbCatalog::open(&path).unwrap());
                next_crash += 1;
            }
            harness.apply(catalog.as_ref().unwrap(), op);
        }

        drop(catalog.take());
        assert_crash_parity(&path, page);
    }
}
