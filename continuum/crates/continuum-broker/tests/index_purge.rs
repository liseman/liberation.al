//! The availability index follows a purge exactly: an index loaded before a purge and then fed each step's delta
//! equals one loaded fresh from the catalog after the purge, whatever the step budgets. This is the permanent check
//! that the broker's index and the catalog agree on what a purge removed (cut frames included).

use continuum_broker::index::{LoadStats, clear_stale_tier0};
use continuum_catalog::{
    Catalog, ChunkFrame, CoverageMark, MemoryCatalog, NewChunk, NewPayload, PurgeProgress, PurgeSelector, StreamInfo,
    StreamKind, frame_slots,
};
use continuum_core::{
    DomainLayer, HeaderBuf, HeaderSpec, Interval, PayloadFormat, StageId, StorageLocator, StorageTier, StreamId, Uuid,
};
use proptest::prelude::*;

/// Loads the index as the broker's startup does: an earlier run's Tier 0 copies left out, and cleared off the catalog.
fn load(
    c: &dyn Catalog,
    page: usize,
) -> Result<(continuum_broker::index::AvailabilityIndex, LoadStats), continuum_catalog::CatalogError> {
    let (index, stats) = continuum_broker::index::load(c, page)?;
    clear_stale_tier0(c, &stats.stale_tier0)?;
    Ok((index, stats))
}

fn id(n: u32) -> Uuid {
    let mut b = [0; 16];
    b[6] = 0x70;
    b[12..].copy_from_slice(&n.to_be_bytes());
    Uuid(b)
}

fn iv(a: u64, b: u64) -> Interval {
    Interval::new(a, b).unwrap()
}

fn info(name: &str, kind: StreamKind) -> StreamInfo {
    StreamInfo {
        name: name.into(),
        kind,
        layer: DomainLayer::Layer1Raw,
        format: PayloadFormat::RawBytes,
        shape_name: String::new(),
        shape_version: 0,
        rate_hz: Some(10.0),
        anchor: None,
    }
}

fn shm(slot: u32) -> StorageLocator {
    StorageLocator::Shm {
        arena: "a".into(),
        arena_generation: 1,
        slab_class: 0,
        slot,
        slot_generation: 1,
        offset: 0,
        length: 1,
    }
}

fn chunk_loc(tier: StorageTier, uri: &str) -> StorageLocator {
    StorageLocator::Chunk { tier, uri: uri.into(), offset: 0, length: 1 }
}

fn payload(n: u32, stream: StreamId, span: (u64, u64), parents: &[u32], subjects: &[u32]) -> HeaderBuf {
    let mut spec = HeaderSpec::raw(id(n), stream, span.0, span.1);
    spec.parents = parents.iter().map(|&p| id(p)).collect();
    spec.subjects = subjects.iter().map(|&s| id(s)).collect();
    HeaderBuf::build(&spec).unwrap()
}

/// cam: frames 1..=10, each 10 ns long from 0; frames 1..=5 are absorbed into chunk 50 (Tier 1) over [0, 50), and
/// 6..=10 stay in Tier 0 (6 and 7 also in Tier 1). derived: a crop 100 + n of each frame (subject 900 on even n),
/// in Tier 1, with coverage of stage 1 on cam for each crop's span. Crops of absorbed frames make those frames keep
/// FRAMES rows, so a purge over them cuts them from the chunk.
fn build(c: &impl Catalog) -> (StreamId, StreamId) {
    let cam = c.register_stream(info("cam", StreamKind::Sensor)).unwrap();
    let derived = c.register_stream(info("crops", StreamKind::Derived)).unwrap();
    for n in 1..=10u32 {
        let t = u64::from(n - 1) * 10;
        c.insert(vec![NewPayload {
            header: payload(n, cam, (t, t + 10), &[], &[]),
            location: Some(shm(n)),
            coverage: None,
        }])
        .unwrap();
        if n == 6 || n == 7 {
            c.set_location(id(n), chunk_loc(StorageTier::Tier1Warm, "frames")).unwrap();
        }
        let subjects: &[u32] = if n % 2 == 0 { &[900] } else { &[] };
        let crop = payload(100 + n, derived, (t, t + 10), &[n], subjects);
        let mark = CoverageMark { source: cam, stage: StageId(1), span: iv(t, t + 10) };
        c.insert(vec![NewPayload {
            header: crop,
            location: Some(chunk_loc(StorageTier::Tier1Warm, "crops")),
            coverage: Some(mark),
        }])
        .unwrap();
    }
    let frames: Vec<ChunkFrame> = (0..5u64).map(|i| ChunkFrame::new(i * 10, i * 100, 100)).collect();
    let chunk = NewChunk {
        payload: NewPayload {
            header: payload(50, cam, (0, 50), &[], &[]),
            location: Some(chunk_loc(StorageTier::Tier1Warm, "chunk")),
            coverage: None,
        },
        frames,
    };
    let absorbed: Vec<Uuid> = (1..=5).map(id).collect();
    c.commit_chunk(chunk, &absorbed).unwrap();
    (cam, derived)
}

#[test]
fn redb_startup_pages_preserve_copies_chunks_and_coverage() {
    let temp = tempfile::tempdir().unwrap();
    let redb = continuum_catalog::RedbCatalog::open(temp.path().join("catalog.redb")).unwrap();
    let memory = MemoryCatalog::new();
    assert_eq!(build(&redb), build(&memory));
    for page in [1, 3, 1024] {
        let (actual, actual_stats) = load(&redb, page).unwrap();
        let (expected, expected_stats) = load(&memory, page).unwrap();
        assert_eq!(actual.snapshot(), expected.snapshot(), "startup page size {page}");
        assert_eq!(actual_stats, expected_stats, "startup counts at page size {page}");
    }
}

/// Steps one purge to its end, feeding each delta to `followed` and, before each step, landing a late insert whose
/// parent is `late_parents[k]` (any id, hidden or not), added to `followed` the way the broker adds a committed
/// ingest. At `restart_at`, a fresh load must equal `followed` (a broker restarting mid-purge).
fn step_purge(
    c: &MemoryCatalog,
    followed: &mut continuum_broker::index::AvailabilityIndex,
    purge: Uuid,
    budgets: &[usize],
    late_parents: &[u32],
    next_late: &mut u32,
    restart_at: Option<usize>,
) -> Result<(), TestCaseError> {
    let derived = c.stream_id("crops").unwrap().unwrap();
    for k in 0.. {
        if let Some(&parent) = late_parents.get(k) {
            let n = *next_late;
            *next_late += 1;
            let t = 2_000 + u64::from(n) * 10;
            let header = payload(n, derived, (t, t + 10), &[parent], &[]);
            let loc = chunk_loc(StorageTier::Tier1Warm, "late");
            // A worker's late result: refused if its parent is gone, hidden if its parent is being purged.
            // The broker adds it to the index only if the catalog didn't hide it as it landed (its parent being
            // purged): a hidden one arrives in a later step's delta.
            if c.insert(vec![NewPayload { header, location: Some(loc), coverage: None }]).is_ok() {
                if c.get(id(n)).unwrap().is_some() {
                    followed.add_copy(derived, iv(t, t + 10), StorageTier::Tier1Warm).unwrap();
                } else {
                    followed.note_hidden_on_landing(id(n));
                }
            }
        }
        if restart_at == Some(k) {
            let (fresh, _) = load(c, 3).unwrap();
            prop_assert_eq!(followed.snapshot(), fresh.snapshot(), "a fresh load mid-purge differs");
        }
        let budget = budgets.get(k).copied().unwrap_or(2).max(1);
        let (delta, done) = match c.purge_step(purge, budget).unwrap() {
            PurgeProgress::Running { delta, .. } => (delta, false),
            PurgeProgress::Removed { delta, .. } => (delta, true),
        };
        prop_assert_eq!(followed.apply_purge_delta(&delta), 0, "a delta entry didn't match the index");
        if done {
            return Ok(());
        }
        prop_assert!(k < 10_000, "the purge never finished");
    }
    Ok(())
}

/// Two purges stepped one after the other (the first can cut frames from the chunk the second dooms whole), with
/// late inserts and an optional mid-purge restart check; afterward the followed index equals a fresh load, copy
/// counts and all.
fn check(
    first: PurgeSelector,
    second: PurgeSelector,
    budgets: &[usize],
    late_parents: &[u32],
    restart_at: Option<usize>,
) -> Result<(), TestCaseError> {
    let c = MemoryCatalog::new();
    build(&c);
    let (mut followed, _) = load(&c, 3).unwrap();
    let mut next_late = 7_000u32;
    c.start_purge(first, "first", "", id(5_000), 1).unwrap();
    step_purge(&c, &mut followed, id(5_000), budgets, late_parents, &mut next_late, restart_at)?;
    c.start_purge(second, "second", "", id(5_001), 2).unwrap();
    step_purge(&c, &mut followed, id(5_001), budgets, late_parents, &mut next_late, None)?;
    let (fresh, _) = load(&c, 4).unwrap();
    prop_assert_eq!(followed.snapshot(), fresh.snapshot());
    Ok(())
}

#[test]
fn a_purge_over_part_of_a_chunk_cuts_its_frames_from_the_index() {
    let c = MemoryCatalog::new();
    let (cam, _) = build(&c);
    let (mut index, stats) = load(&c, 3).unwrap();
    assert_eq!(stats.streams, 2);
    let before = index.covered(cam, iv(0, 100));
    assert_eq!(before.first().map(|c| c.interval.start_ns()), Some(0));
    let report = {
        c.start_purge(PurgeSelector::Range { span: iv(15, 25), streams: Some(vec![cam]) }, "t", "", id(6_000), 1)
            .unwrap();
        loop {
            match c.purge_step(id(6_000), 2).unwrap() {
                PurgeProgress::Running { delta, .. } => assert_eq!(index.apply_purge_delta(&delta), 0),
                PurgeProgress::Removed { delta, report } => {
                    assert_eq!(index.apply_purge_delta(&delta), 0);
                    break report;
                }
            }
        }
    };
    assert!(!report.removed.is_empty());
    // [15, 25) overlaps the slots of the frames at 10 and 20, so both are cut from the chunk, the same as they would
    // be before compaction: [10, 30) isn't available on cam any more.
    for c in index.covered(cam, iv(0, 100)) {
        assert!(!c.interval.overlaps(iv(10, 30)), "{c:?} still covers purged time");
    }
}

#[test]
fn a_chunk_cut_by_one_purge_and_doomed_whole_by_another_leaves_no_drift() {
    let c = MemoryCatalog::new();
    let (cam, _) = build(&c);
    let (mut index, _) = load(&c, 3).unwrap();
    let mut next = 7_000;
    c.start_purge(PurgeSelector::Range { span: iv(10, 30), streams: Some(vec![cam]) }, "a", "", id(5_000), 1).unwrap();
    step_purge(&c, &mut index, id(5_000), &[], &[], &mut next, None).unwrap();
    // The whole chunk now: its full span is reported, and the index must remove only what it still held.
    c.start_purge(PurgeSelector::Range { span: iv(0, 50), streams: Some(vec![cam]) }, "b", "", id(5_001), 2).unwrap();
    step_purge(&c, &mut index, id(5_001), &[], &[], &mut next, None).unwrap();
    let (fresh, _) = load(&c, 4).unwrap();
    assert_eq!(index.snapshot(), fresh.snapshot());
}

fn selector() -> impl Strategy<Value = (bool, u64, u64)> {
    (any::<bool>(), 0u64..90, 1u64..40)
}

fn to_selector((entity, start, len): (bool, u64, u64), cam: StreamId) -> PurgeSelector {
    if entity {
        PurgeSelector::Entity(id(900))
    } else {
        PurgeSelector::Range { span: iv(start, start + len), streams: Some(vec![cam]) }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn purges_followed_by_deltas_match_a_fresh_load(
        first in selector(),
        second in selector(),
        budgets in prop::collection::vec(1usize..5, 0..30),
        late_parents in prop::collection::vec(prop_oneof![1u32..=10, 101u32..=110], 0..8),
        restart_at in proptest::option::of(0usize..6),
    ) {
        // The stream ids build() assigns are the same in every fresh MemoryCatalog.
        let cam = { let c = MemoryCatalog::new(); build(&c).0 };
        check(to_selector(first, cam), to_selector(second, cam), &budgets, &late_parents, restart_at)?;
    }
}

/// The chunk the tier worker commits over frames 6..=10 ([50, 100)), and what the broker does with the commit's
/// answer: the copy over the span, less the slots the chunk landed without.
fn commit_late_chunk(
    c: &MemoryCatalog,
    index: &mut continuum_broker::index::AvailabilityIndex,
    cam: StreamId,
) -> Vec<u32> {
    let frames: Vec<ChunkFrame> = (0..5u64).map(|i| ChunkFrame::new(50 + i * 10, i * 100, 100)).collect();
    let chunk = NewChunk {
        payload: NewPayload {
            header: payload(51, cam, (50, 100), &[], &[]),
            location: Some(chunk_loc(StorageTier::Tier1Warm, "chunk2")),
            coverage: None,
        },
        frames: frames.clone(),
    };
    let inserted = c.commit_chunk(chunk, &(6..=10).map(id).collect::<Vec<_>>()).unwrap();
    assert!(!inserted.hidden && !inserted.duplicate);
    let slots = frame_slots(iv(50, 100), &frames);
    let cut: Vec<Interval> = inserted.cut.iter().map(|&i| slots[i as usize]).collect();
    index.add_chunk_copy(id(51), cam, iv(50, 100), StorageTier::Tier1Warm, &cut).unwrap();
    inserted.cut
}

fn run_purge(c: &MemoryCatalog, index: &mut continuum_broker::index::AvailabilityIndex, purge: Uuid) {
    loop {
        match c.purge_step(purge, 1).unwrap() {
            PurgeProgress::Running { delta, .. } => assert_eq!(index.apply_purge_delta(&delta), 0),
            PurgeProgress::Removed { delta, .. } => {
                assert_eq!(index.apply_purge_delta(&delta), 0);
                return;
            }
        }
    }
}

#[test]
fn a_chunk_committed_over_a_frame_a_finished_purge_removed_lands_without_that_slot() {
    let c = MemoryCatalog::new();
    let (cam, _) = build(&c);
    let (mut index, _) = load(&c, 3).unwrap();
    // The purge of [60, 70) is done (frame 7 is gone, with its crop) when the tier worker commits the chunk that
    // absorbed frames 6..=10: the chunk lands without frame 7's slot, and nothing will report that cut again.
    c.start_purge(PurgeSelector::Range { span: iv(60, 70), streams: Some(vec![cam]) }, "t", "", id(6_001), 1).unwrap();
    run_purge(&c, &mut index, id(6_001));
    assert_eq!(commit_late_chunk(&c, &mut index, cam), vec![1]);
    // (The absorbed frames' Tier 0 copies stay in the index until their slots are reclaimed, so a fresh load, which
    // no longer sees those frames, isn't the comparison here.)
    for covered in index.covered(cam, iv(50, 100)) {
        assert!(!covered.interval.overlaps(iv(60, 70)), "{covered:?} quotes the purged frame's slot from the chunk");
    }
    assert!(index.snapshot().cuts.contains(&(id(51).0, vec![iv(60, 70)])), "the chunk's cut slot is on record");
}

#[test]
fn a_chunk_committed_over_a_frame_a_running_purge_hid_loses_that_slot_in_the_purges_delta() {
    let c = MemoryCatalog::new();
    let (cam, _) = build(&c);
    let (mut index, _) = load(&c, 3).unwrap();
    // The purge of [60, 70) has hidden frame 7 (one step) when the chunk commits: the commit names no cut (the purge
    // reports it), the index gets the whole copy, and the purge's next delta takes the slot out, so following the
    // deltas and loading fresh agree all the way.
    c.start_purge(PurgeSelector::Range { span: iv(60, 70), streams: Some(vec![cam]) }, "t", "", id(6_002), 1).unwrap();
    match c.purge_step(id(6_002), 1).unwrap() {
        PurgeProgress::Running { delta, .. } => assert_eq!(index.apply_purge_delta(&delta), 0),
        PurgeProgress::Removed { .. } => panic!("finished in one step"),
    }
    assert_eq!(commit_late_chunk(&c, &mut index, cam), Vec::<u32>::new());
    let over_slot = |index: &continuum_broker::index::AvailabilityIndex| {
        index.covered(cam, iv(60, 70)).into_iter().map(|c| (c.interval, c.tier)).collect::<Vec<_>>()
    };
    assert_eq!(over_slot(&index), vec![(iv(60, 70), StorageTier::Tier1Warm)], "the whole copy, until the delta");
    run_purge(&c, &mut index, id(6_002));
    assert_eq!(over_slot(&index), vec![], "the delta's cut took the slot out of the chunk's copy");
    assert!(index.snapshot().cuts.contains(&(id(51).0, vec![iv(60, 70)])), "the chunk's cut slot is on record");
}

#[test]
fn a_load_leaves_out_an_earlier_runs_tier0_copies_and_names_their_payloads() {
    let c = MemoryCatalog::new();
    let (cam, _) = build(&c);
    let (index, stats) = load(&c, 3).unwrap();
    // Frames 6..=10 were left in Tier 0, the last run's arena, which is gone: 6 and 7 are still in Tier 1, but
    // 8..=10 ([70, 100)) aren't anywhere now, so the index must not call them hot.
    let covered: Vec<(Interval, StorageTier)> =
        index.covered(cam, iv(50, 100)).iter().map(|c| (c.interval, c.tier)).collect();
    assert_eq!(covered, [(iv(50, 70), StorageTier::Tier1Warm)]);
    let mut stale = stats.stale_tier0.clone();
    stale.sort();
    assert_eq!(stale, (6..=10).map(id).collect::<Vec<_>>());
    // Startup cleared their rows, so the next run finds none.
    assert!(load(&c, 3).unwrap().1.stale_tier0.is_empty());
}
