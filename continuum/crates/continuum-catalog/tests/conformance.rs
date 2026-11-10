//! The [`Catalog`] contract, as tests generic over the implementation. Each implementation runs all of them through
//! `conformance!` at the bottom, so the in-memory and redb catalogs can't drift apart.

use std::sync::Arc;

use continuum_catalog::*;
use continuum_core::{
    BootId, DomainLayer, HeaderBuf, HeaderSpec, Interval, PayloadFormat, StageId, StorageLocator, StorageTier,
    StreamId, Uuid,
};

fn id(n: u16) -> Uuid {
    let mut b = [0; 16];
    b[14..].copy_from_slice(&n.to_be_bytes());
    Uuid(b)
}

fn iv(s: u64, e: u64) -> Interval {
    Interval::new(s, e).unwrap()
}

fn info(name: &str) -> StreamInfo {
    StreamInfo {
        name: name.into(),
        kind: StreamKind::Sensor,
        layer: DomainLayer::Layer1Raw,
        format: PayloadFormat::RawBytes,
        shape_name: "NV12".into(),
        shape_version: 1,
        rate_hz: Some(30.0),
        anchor: None,
    }
}

fn header(n: u16, stream: StreamId, start: u64, end: u64, seq: u32, parents: &[u16], subjects: &[u16]) -> HeaderBuf {
    let mut spec = HeaderSpec::raw(id(n), stream, start, end);
    spec.seq = seq;
    spec.parents = parents.iter().map(|&p| id(p)).collect();
    spec.subjects = subjects.iter().map(|&s| id(s)).collect();
    HeaderBuf::build(&spec).unwrap()
}

fn shm(slot: u32) -> StorageLocator {
    StorageLocator::Shm {
        arena: "a".into(),
        arena_generation: 1,
        slab_class: 0,
        slot,
        slot_generation: 1,
        offset: u64::from(slot) * 16_384,
        length: 100,
    }
}

fn warm(uri: &str) -> StorageLocator {
    StorageLocator::Chunk { tier: StorageTier::Tier1Warm, uri: uri.into(), offset: 0, length: 1000 }
}

fn new(h: HeaderBuf) -> NewPayload {
    NewPayload { header: h, location: None, coverage: None }
}

fn ids(page: &Page) -> Vec<Uuid> {
    page.items.iter().map(|i| i.header.id()).collect()
}

fn present<C: Catalog>(c: &C, n: u16) -> bool {
    c.get(id(n)).unwrap().is_some()
}

fn an_age_out_removes_only_what_it_names_and_records_its_spans<C: Catalog>(c: C) {
    let (cam, d) = graph(&c);
    c.add_coverage(CoverageMark { source: cam, stage: StageId(7), span: iv(0, 50) }).unwrap();
    c.mark_commit_clock(100).unwrap();
    let by = c.commit_seq_at(100).unwrap();
    assert_eq!(by, c.head_seq().unwrap());
    let report = c
        .purge(
            PurgeSelector::AgeOut { ids: vec![id(1), id(11)], committed_by_seq: by, merge_gap_ns: 0 },
            "aged",
            "",
            id(700),
            5,
        )
        .unwrap();
    let mut gone: Vec<Uuid> = report.removed.iter().map(|r| r.id).collect();
    gone.sort();
    assert_eq!(gone, vec![id(1), id(11)], "nothing derived from them goes");
    for n in [21, 31, 2, 12] {
        assert!(present(&c, n), "{n} stays");
    }
    // Its header still names the aged-out parent, but the parent index forgets an id that no longer resolves.
    let Some(Resolved::Payload(child)) = c.get(id(21)).unwrap() else { panic!("21 stays") };
    assert_eq!(child.header.parents(), vec![id(11)], "the child's header still names its aged-out parent");
    assert_eq!(c.parents(id(21)).unwrap(), Vec::<Uuid>::new(), "the parent index drops it");
    assert_eq!(c.parents(id(31)).unwrap(), vec![id(21), id(22)], "parents that are still there stay");
    assert_eq!(c.coverage(cam, StageId(7), iv(0, 50)).unwrap(), vec![iv(0, 50)], "coverage stays: it was processed");
    assert_eq!(c.aged_out(cam, iv(0, 100), 10).unwrap(), vec![iv(10, 11)]);
    assert_eq!(c.aged_out(d, iv(0, 100), 10).unwrap(), vec![iv(10, 11)]);
    assert!(c.aged_out(cam, iv(20, 100), 10).unwrap().is_empty());
    let record = c.purge_record(id(700)).unwrap().unwrap();
    assert_eq!(record.kind, PurgeKind::AgeOut { count: 2 });
    assert!(matches!(c.aged_out(cam, iv(0, 1), 0), Err(CatalogError::Invalid(_))));
    // Forgetting still reaches what was derived from the aged-out rows, through what remains.
    let report = c.purge(PurgeSelector::Ids(vec![id(21)]), "forget", "", id(701), 5).unwrap();
    let mut gone: Vec<Uuid> = report.removed.iter().map(|r| r.id).collect();
    gone.sort();
    assert_eq!(gone, vec![id(21), id(31)]);
    assert!(present(&c, 22) && present(&c, 12), "the other branch stays");
}

fn an_age_out_with_a_gap_records_runs_and_keeps_wider_gaps_apart<C: Catalog>(c: C) {
    let (cam, _) = graph(&c);
    c.mark_commit_clock(100).unwrap();
    let by = c.commit_seq_at(100).unwrap();
    // Instants at 10, 20, and 30, each [t, t + 1): 9 ns apart, within a 10 ns gap.
    let ids = vec![id(1), id(2), id(3)];
    c.purge(PurgeSelector::AgeOut { ids, committed_by_seq: by, merge_gap_ns: 10 }, "aged", "", id(710), 5).unwrap();
    assert_eq!(c.aged_out(cam, iv(0, 100), 10).unwrap(), vec![iv(10, 31)], "one run, not three entries");
    // The one at 40 is 9 ns past the run: a 5 ns gap keeps it apart, so the time between still reads as not aged.
    c.purge(PurgeSelector::AgeOut { ids: vec![id(4)], committed_by_seq: by, merge_gap_ns: 5 }, "aged", "", id(711), 5)
        .unwrap();
    assert_eq!(c.aged_out(cam, iv(0, 100), 10).unwrap(), vec![iv(10, 31), iv(40, 41)]);
}

fn a_wider_gap_joins_runs_left_by_narrower_ones_end_to_end<C: Catalog>(c: C) {
    let (cam, _) = graph(&c);
    c.mark_commit_clock(100).unwrap();
    let by = c.commit_seq_at(100).unwrap();
    // A case from review: [10, 11), [20, 21), and [30, 31) aged one at a time with no gap, then [0, 1) with a gap of
    // 10.
    for (n, purge) in [(2, 730), (3, 731), (4, 732)] {
        let exact = PurgeSelector::AgeOut { ids: vec![id(n)], committed_by_seq: by, merge_gap_ns: 0 };
        c.purge(exact, "aged", "", id(purge), 5).unwrap();
    }
    assert_eq!(c.aged_out(cam, iv(0, 100), 10).unwrap(), vec![iv(20, 21), iv(30, 31), iv(40, 41)]);
    c.purge(PurgeSelector::AgeOut { ids: vec![id(1)], committed_by_seq: by, merge_gap_ns: 10 }, "aged", "", id(733), 5)
        .unwrap();
    assert_eq!(c.aged_out(cam, iv(0, 100), 10).unwrap(), vec![iv(10, 41)], "each span absorbed brings the next");
}

fn a_completed_age_out_keeps_counting_what_it_removed<C: Catalog>(c: C) {
    let (_, _) = graph(&c);
    c.mark_commit_clock(100).unwrap();
    let by = c.commit_seq_at(100).unwrap();
    // id 999 doesn't exist: asked for three, removed two.
    let aged = PurgeSelector::AgeOut { ids: vec![id(1), id(2), id(999)], committed_by_seq: by, merge_gap_ns: 0 };
    c.purge(aged, "aged", "", id(740), 5).unwrap();
    let removed = |c: &C| c.purges(None, 100).unwrap().into_iter().find(|p| p.purge_id == id(740)).unwrap().removed;
    assert_eq!(removed(&c), 2);
    c.complete_purge(id(740), 1_000).unwrap();
    assert_eq!(removed(&c), 2, "the collapsed report still says how many went");
}

fn without_a_gap_each_aged_payload_keeps_its_own_span<C: Catalog>(c: C) {
    let (cam, _) = graph(&c);
    c.mark_commit_clock(100).unwrap();
    let by = c.commit_seq_at(100).unwrap();
    let ids = vec![id(1), id(2)];
    c.purge(PurgeSelector::AgeOut { ids, committed_by_seq: by, merge_gap_ns: 0 }, "aged", "", id(712), 5).unwrap();
    assert_eq!(c.aged_out(cam, iv(0, 100), 10).unwrap(), vec![iv(10, 11), iv(20, 21)]);
}

fn a_completed_age_out_keeps_its_spans_and_count_but_not_its_items<C: Catalog>(c: C) {
    let (cam, d) = graph(&c);
    c.mark_commit_clock(100).unwrap();
    let by = c.commit_seq_at(100).unwrap();
    let aged = PurgeSelector::AgeOut { ids: vec![id(1), id(2)], committed_by_seq: by, merge_gap_ns: 10 };
    c.purge(aged, "aged", "", id(720), 5).unwrap();
    let before = c.purge_record(id(720)).unwrap().unwrap();
    assert_eq!(before.report.unwrap().removed.len(), 2, "the items stay until every keeper has its notice");
    c.complete_purge(id(720), 1_000).unwrap();
    let after = c.purge_record(id(720)).unwrap().unwrap();
    assert_eq!(after.kind, PurgeKind::AgeOut { count: 2 }, "the count stays");
    assert_eq!(after.completed_ns, Some(1_000));
    assert!(after.report.unwrap().removed.is_empty(), "the items collapse to the spans");
    assert_eq!(c.aged_out(cam, iv(0, 100), 10).unwrap(), vec![iv(10, 21)], "the spans answer what was aged");
    // A forget or delete keeps its exact report for good.
    c.purge(PurgeSelector::Ids(vec![id(12)]), "deleted", "", id(721), 5).unwrap();
    c.complete_purge(id(721), 1_001).unwrap();
    let kept = c.purge_record(id(721)).unwrap().unwrap().report.unwrap();
    assert!(kept.removed.iter().any(|r| r.id == id(12) && r.stream == d), "a delete's items stay");
}

/// A stretch record on its own stream, depending on `deps` (stream, start, end), each at its stream's current revision.
fn stretch<C: Catalog>(c: &C, n: u16, on: StreamId, at: u64, deps: &[(StreamId, u64, u64)]) {
    let deps: Vec<Dependency> = deps
        .iter()
        .map(|&(stream, s, e)| Dependency {
            stream,
            span: iv(s, e),
            revision: c.invalidation_revision(stream).unwrap().unwrap(),
        })
        .collect();
    c.insert_dependent(vec![new(header(n, on, at, at, 0, &[], &[]))], vec![(id(n), deps)]).unwrap();
}

fn a_purge_takes_what_depends_on_a_stretch_it_removed_and_what_derives_from_that<C: Catalog>(c: C) {
    let (cam, d) = graph(&c);
    let progress = c.register_stream(info("progress")).unwrap();
    stretch(&c, 600, progress, 5, &[(cam, 0, 25)]); // the frames at 10 and 20
    stretch(&c, 601, progress, 6, &[(cam, 30, 50)]); // the frames at 30 and 40
    stretch(&c, 602, progress, 7, &[(d, 10, 11)]); // an output stream's stretch: 11 and 21 lie in it
    c.insert(vec![new(header(603, progress, 8, 8, 1, &[600], &[]))]).unwrap(); // derived from 600
    // A purge of an output in an output stream's stretch takes that stretch, though the video stays.
    c.purge(PurgeSelector::Ids(vec![id(21)]), "privacy", "", id(800), 50).unwrap();
    assert!(!present(&c, 602));
    assert!(present(&c, 600) && present(&c, 601) && present(&c, 1) && present(&c, 2));
    // An id purge of the frame at 20 takes 600 (and 603, derived from it), not 601.
    c.purge(PurgeSelector::Ids(vec![id(2)]), "privacy", "", id(801), 50).unwrap();
    assert!(!present(&c, 600) && !present(&c, 603), "the stretch over it, and what came of that, went");
    assert!(present(&c, 601) && present(&c, 3) && present(&c, 4));
    // A range purge over 35..45 removes the frame at 40, so 601 goes, found by its dependency, not its own time.
    c.purge(PurgeSelector::Range { span: iv(35, 45), streams: Some(vec![cam]) }, "privacy", "", id(802), 50).unwrap();
    assert!(!present(&c, 601));
}

fn dependencies_with_one_start_keep_the_longest_reach<C: Catalog>(c: C) {
    let (cam, _) = graph(&c);
    let progress = c.register_stream(info("progress")).unwrap();
    // Two dependencies on one stream from one start, the shorter named last: it mustn't replace the longer.
    stretch(&c, 620, progress, 5, &[(cam, 30, 50), (cam, 30, 35)]);
    // A repeat naming them in the other order is the same set.
    let rev = c.invalidation_revision(cam).unwrap().unwrap();
    let again = [(30, 35), (30, 50)].map(|(s, e)| Dependency { stream: cam, span: iv(s, e), revision: rev }).to_vec();
    let repeat = c.insert_dependent(vec![new(header(620, progress, 5, 5, 0, &[], &[]))], vec![(id(620), again)]);
    assert!(repeat.unwrap()[0].duplicate);
    // The frame at 40 lies only in the longer one.
    c.purge(PurgeSelector::Range { span: iv(40, 45), streams: Some(vec![cam]) }, "privacy", "", id(820), 50).unwrap();
    assert!(!present(&c, 620), "the longer dependency's reach took it");
}

fn a_cut_frame_takes_what_depends_on_its_slot<C: Catalog>(c: C) {
    let s = chunked(&c);
    let progress = c.register_stream(info("progress")).unwrap();
    stretch(&c, 610, progress, 5, &[(s, 30, 40)]); // inside the frame at 33's slot
    stretch(&c, 611, progress, 6, &[(s, 70, 90)]); // inside the frame at 66's slot
    c.purge(PurgeSelector::Range { span: iv(33, 34), streams: Some(vec![s]) }, "privacy", "", id(810), 50).unwrap();
    assert!(!present(&c, 610), "the cut frame's dependents went with it");
    assert!(present(&c, 611) && present(&c, 100), "the chunk and the rest stay");
}

fn an_age_out_leaves_what_depends_on_what_it_aged<C: Catalog>(c: C) {
    let (cam, _) = graph(&c);
    let progress = c.register_stream(info("progress")).unwrap();
    stretch(&c, 620, progress, 5, &[(cam, 0, 25)]);
    c.mark_commit_clock(100).unwrap();
    let by = c.commit_seq_at(100).unwrap();
    c.purge(
        PurgeSelector::AgeOut { ids: vec![id(1), id(2)], committed_by_seq: by, merge_gap_ns: 0 },
        "aged",
        "",
        id(820),
        50,
    )
    .unwrap();
    assert!(present(&c, 620), "aging the video never removes a stretch record");
    // Gone itself, a dependent leaves no dependency behind for a later purge to find.
    c.purge(PurgeSelector::Ids(vec![id(620)]), "deleted", "", id(821), 50).unwrap();
    let report = c
        .purge(PurgeSelector::Range { span: iv(0, 100), streams: Some(vec![cam]) }, "privacy", "", id(822), 50)
        .unwrap();
    assert!(report.removed.iter().all(|r| r.id != id(620)));
}

fn a_repeat_must_name_the_dependencies_it_was_indexed_with<C: Catalog>(c: C) {
    let (cam, _) = graph(&c);
    let progress = c.register_stream(info("progress")).unwrap();
    let dep = |s: u64, e: u64| Dependency {
        stream: cam,
        span: iv(s, e),
        revision: c.invalidation_revision(cam).unwrap().unwrap(),
    };
    let row = || new(header(640, progress, 5, 5, 0, &[], &[]));
    let first = c.insert_dependent(vec![row()], vec![(id(640), vec![dep(0, 25)])]).unwrap();
    let again = c.insert_dependent(vec![row()], vec![(id(640), vec![dep(0, 25)])]).unwrap();
    assert!(again[0].duplicate && again[0].seq == first[0].seq, "the same dependencies: a repeat");
    for (label, r) in
        [("none", c.insert(vec![row()])), ("other", c.insert_dependent(vec![row()], vec![(id(640), vec![dep(0, 30)])]))]
    {
        assert!(matches!(r, Err(CatalogError::Conflict(_))), "{label}: {r:?}");
    }
    // A payload indexed without dependencies repeats as before.
    c.insert(vec![new(header(641, progress, 6, 6, 1, &[], &[]))]).unwrap();
    assert!(c.insert(vec![new(header(641, progress, 6, 6, 1, &[], &[]))]).unwrap()[0].duplicate);
}

fn a_chunk_never_absorbs_a_payload_with_dependencies<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    let rev = c.invalidation_revision(s).unwrap().unwrap();
    // A frame indexed with a dependency (as a derived stretch would be): absorbing it would drop its header.
    let frame = NewPayload { header: header(1, s, 0, 0, 0, &[], &[]), location: Some(shm(1)), coverage: None };
    let dep = Dependency { stream: s, span: iv(0, 10), revision: rev };
    c.insert_dependent(vec![frame], vec![(id(1), vec![dep])]).unwrap();
    let chunk = NewChunk {
        payload: NewPayload {
            header: header(100, s, 0, 33, 0, &[], &[]),
            location: Some(warm("file:///c/9")),
            coverage: None,
        },
        frames: vec![ChunkFrame::new(0, 0, 300)],
    };
    let refused = c.commit_chunk(chunk, &[id(1)]);
    assert!(matches!(refused, Err(CatalogError::Invalid(ref why)) if why.contains("dependencies")), "{refused:?}");
    assert!(present(&c, 1), "the dependent keeps its own row");
}

fn a_dependency_read_before_a_purge_is_refused<C: Catalog>(c: C) {
    let (cam, _) = graph(&c);
    let progress = c.register_stream(info("progress")).unwrap();
    let read = c.invalidation_revision(cam).unwrap().unwrap();
    // A purge lands between the producer's check and its publication.
    c.purge(PurgeSelector::Ids(vec![id(2)]), "privacy", "", id(830), 50).unwrap();
    let stale = Dependency { stream: cam, span: iv(0, 25), revision: read };
    let refused = c.insert_dependent(vec![new(header(630, progress, 5, 5, 0, &[], &[]))], vec![(id(630), vec![stale])]);
    assert!(matches!(refused, Err(CatalogError::Conflict(_))), "{refused:?}");
    assert!(!present(&c, 630), "refused whole: never indexed without its dependency");
    let unknown = Dependency { stream: StreamId(9999), span: iv(0, 1), revision: 0 };
    let r = c.insert_dependent(vec![new(header(631, progress, 5, 5, 1, &[], &[]))], vec![(id(631), vec![unknown])]);
    assert!(matches!(r, Err(CatalogError::NotFound(_))), "{r:?}");
    let stray = Dependency { stream: cam, span: iv(0, 1), revision: c.invalidation_revision(cam).unwrap().unwrap() };
    let r = c.insert_dependent(vec![new(header(632, progress, 5, 5, 2, &[], &[]))], vec![(id(999), vec![stray])]);
    assert!(matches!(r, Err(CatalogError::Invalid(_))), "{r:?}");
}

fn an_age_out_skips_what_was_committed_after_its_cutoff<C: Catalog>(c: C) {
    let (cam, _) = graph(&c);
    assert_eq!(c.commit_seq_at(100).unwrap(), 0, "no mark that old yet");
    c.mark_commit_clock(100).unwrap();
    let by = c.commit_seq_at(150).unwrap();
    // Backdated: an import commits media from long ago, after the cutoff.
    c.insert(vec![new(header(5, cam, 5, 5, 0, &[], &[]))]).unwrap();
    c.mark_commit_clock(90).unwrap(); // the clock never goes back
    assert_eq!(c.commit_seq_at(95).unwrap(), 0);
    c.mark_commit_clock(200).unwrap();
    assert_eq!(c.commit_seq_at(200).unwrap(), c.head_seq().unwrap());
    assert_eq!(c.commit_seq_at(150).unwrap(), by);
    let report = c
        .purge(
            PurgeSelector::AgeOut { ids: vec![id(5), id(1), id(999)], committed_by_seq: by, merge_gap_ns: 0 },
            "aged",
            "",
            id(701),
            5,
        )
        .unwrap();
    assert_eq!(report.removed.iter().map(|r| r.id).collect::<Vec<_>>(), vec![id(1)]);
    assert!(present(&c, 5), "committed after the cutoff, whatever its own time");
}

fn a_child_landing_under_an_age_out_is_not_hidden<C: Catalog>(c: C) {
    let (_, d) = graph(&c);
    c.mark_commit_clock(100).unwrap();
    let by = c.commit_seq_at(100).unwrap();
    c.start_purge(
        PurgeSelector::AgeOut { ids: vec![id(1)], committed_by_seq: by, merge_gap_ns: 0 },
        "aged",
        "",
        id(702),
        5,
    )
    .unwrap();
    c.purge_step(id(702), 1).unwrap();
    let landed = c.insert(vec![new(header(40, d, 10, 10, 9, &[1], &[]))]).unwrap();
    assert!(!landed[0].hidden, "an age-out takes only what it names");
    while !matches!(c.purge_step(id(702), 1).unwrap(), PurgeProgress::Removed { .. }) {}
    assert!(present(&c, 40) && !present(&c, 1));
}

fn a_chunk_aged_out_whole_keeps_what_its_frames_led_to<C: Catalog>(c: C) {
    chunked(&c);
    c.mark_commit_clock(1_000).unwrap();
    let by = c.commit_seq_at(1_000).unwrap();
    let report = c
        .purge(
            PurgeSelector::AgeOut { ids: vec![id(100)], committed_by_seq: by, merge_gap_ns: 0 },
            "aged",
            "",
            id(703),
            5,
        )
        .unwrap();
    assert_eq!(report.removed.iter().map(|r| r.id).collect::<Vec<_>>(), vec![id(100)]);
    let mut frames = report.removed_frames.clone();
    frames.sort();
    assert_eq!(frames, vec![id(1), id(2), id(3)], "its frames' rows go with the chunk (none pruned yet)");
    assert!(present(&c, 10), "the crop derived from frame 2 stays");
    let Some(Resolved::Payload(crop)) = c.get(id(10)).unwrap() else { panic!("the crop stays") };
    assert_eq!(crop.header.parents(), vec![id(2)], "its header still names the frame");
    assert_eq!(c.parents(id(10)).unwrap(), Vec::<Uuid>::new(), "the frame's row is gone, so the index forgets it");
}

fn recorded_aged_out_spans_merge_and_need_a_known_stream<C: Catalog>(c: C) {
    let s = chunked(&c);
    c.record_aged_out(s, &[iv(33, 99)]).unwrap();
    c.record_aged_out(s, &[iv(200, 300), iv(99, 120)]).unwrap();
    assert_eq!(c.aged_out(s, iv(0, 1_000), 10).unwrap(), vec![iv(33, 120), iv(200, 300)]);
    assert!(matches!(c.record_aged_out(StreamId(u32::MAX), &[iv(0, 1)]), Err(CatalogError::NotFound(_))));
}

/// One write looks at frames until its window says stop, and the next picks up after the last one it looked
/// at, so a sweep's commit covers as many frames as its time allows rather than one.
#[test]
fn redb_a_prune_window_stops_when_told_and_resumes_after_the_last_frame_looked_at() {
    let dir = tempfile::tempdir().unwrap();
    let c = RedbCatalog::open(dir.path().join("prune.redb")).unwrap();
    chunked(&c);
    let mut asked = 0;
    let first = c
        .prune_frames_until(100, None, 64, &mut || {
            asked += 1;
            asked >= 2 // two frames looked at, then stop
        })
        .unwrap();
    assert_eq!(first.scanned, 2, "it stops once told, not at the row ceiling");
    let rest = c.prune_frames_until(100, first.next, 64, &mut || false).unwrap();
    assert_eq!((rest.scanned, rest.next), (1, None), "the next window looks at the one frame left and ends the sweep");
    assert_eq!(first.pruned + rest.pruned, 2, "the same frames go as in one sweep: 1 and 3, not 2, which has a child");
    assert!(!present(&c, 1) && present(&c, 2) && !present(&c, 3));
}

#[test]
fn redb_stream_generation_survives_a_restart_and_a_scrub() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("generation.redb");
    let c = RedbCatalog::open(&path).unwrap();
    c.register_stream(info("a")).unwrap();
    c.register_stream(info("b")).unwrap();
    assert_eq!(c.stream_generation().unwrap(), Some(2));
    while !matches!(c.scrub_step(64).unwrap(), ScrubProgress::Idle | ScrubProgress::Swapped) {}
    drop(c);
    let c = RedbCatalog::open(&path).unwrap();
    assert_eq!(c.stream_generation().unwrap(), Some(2));
    c.register_stream(info("c")).unwrap();
    assert_eq!(c.stream_generation().unwrap(), Some(3));
}

#[test]
fn redb_aging_state_survives_a_restart_and_a_scrub() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("aging.redb");
    let c = RedbCatalog::open(&path).unwrap();
    let (cam, _) = graph(&c);
    c.mark_commit_clock(100).unwrap();
    let by = c.commit_seq_at(100).unwrap();
    c.start_purge(
        PurgeSelector::AgeOut { ids: vec![id(1), id(2)], committed_by_seq: by, merge_gap_ns: 0 },
        "aged",
        "",
        id(704),
        5,
    )
    .unwrap();
    c.purge_step(id(704), 1).unwrap();
    drop(c);
    let c = RedbCatalog::open(&path).unwrap();
    while !matches!(c.purge_step(id(704), 1).unwrap(), PurgeProgress::Removed { .. }) {}
    assert!(present(&c, 11) && !present(&c, 1) && !present(&c, 2));
    while !matches!(c.scrub_step(64).unwrap(), ScrubProgress::Idle | ScrubProgress::Swapped) {}
    drop(c);
    let c = RedbCatalog::open(&path).unwrap();
    assert_eq!(c.aged_out(cam, iv(0, 100), 10).unwrap(), vec![iv(10, 11), iv(20, 21)]);
    assert_eq!(c.commit_seq_at(100).unwrap(), by);
}

fn invalidation_revisions_follow_cross_stream_deletions_and_frame_cuts<C: Catalog>(c: C) {
    let (raw, derived) = graph(&c);
    let unrelated = c.register_stream(info("unrelated")).unwrap();
    assert_eq!(c.invalidation_revision(raw).unwrap(), Some(0));
    assert_eq!(c.invalidation_revision(derived).unwrap(), Some(0));
    assert!(matches!(c.invalidation_revision(StreamId(u32::MAX)), Err(CatalogError::NotFound(_))));
    c.start_purge(PurgeSelector::Ids(vec![id(1)]), "test", "", id(600), 1).unwrap();
    let mut previous = 0;
    loop {
        let progress = c.purge_step(id(600), 1).unwrap();
        let revision = c.invalidation_revision(raw).unwrap().unwrap();
        assert!(revision >= previous);
        if !present(&c, 1) {
            assert!(revision > 0);
        }
        previous = revision;
        if matches!(progress, PurgeProgress::Removed { .. }) {
            break;
        }
    }
    assert!(c.invalidation_revision(derived).unwrap().unwrap() > 0);
    assert_eq!(c.invalidation_revision(unrelated).unwrap(), Some(0));
    let before = c.invalidation_revision(derived).unwrap();
    c.insert(vec![new(header(80, derived, 90, 91, 0, &[], &[]))]).unwrap();
    assert_eq!(c.invalidation_revision(derived).unwrap(), before, "append uses commit sequence");
}

fn invalidation_revision_advances_when_only_a_chunk_frame_is_cut<C: Catalog>(c: C) {
    let stream = chunked(&c);
    let before = c.invalidation_revision(stream).unwrap().unwrap();
    c.start_purge(PurgeSelector::Range { span: iv(35, 40), streams: Some(vec![stream]) }, "test", "", id(601), 1)
        .unwrap();
    loop {
        let progress = c.purge_step(id(601), 1).unwrap();
        if matches!(c.get(id(100)).unwrap(), Some(Resolved::Payload(p)) if p.frames.as_ref().unwrap()[1].cut) {
            assert!(c.invalidation_revision(stream).unwrap().unwrap() > before);
            break;
        }
        assert!(!matches!(progress, PurgeProgress::Removed { .. }));
    }
}

#[test]
fn redb_invalidation_revision_survives_pending_restart_and_scrub() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("revision.redb");
    let c = RedbCatalog::open(&path).unwrap();
    let (raw, derived) = graph(&c);
    c.start_purge(PurgeSelector::Ids(vec![id(1)]), "test", "", id(602), 1).unwrap();
    while present(&c, 1) {
        c.purge_step(id(602), 1).unwrap();
    }
    let before = c.invalidation_revision(raw).unwrap().unwrap();
    assert!(before > 0);
    drop(c);
    let c = RedbCatalog::open(&path).unwrap();
    assert_eq!(c.invalidation_revision(raw).unwrap(), Some(before));
    while !matches!(c.purge_step(id(602), 1).unwrap(), PurgeProgress::Removed { .. }) {}
    assert!(c.invalidation_revision(derived).unwrap().unwrap() > 0);
    let finished = c.invalidation_revision(raw).unwrap();
    while !matches!(c.scrub_step(64).unwrap(), ScrubProgress::Idle | ScrubProgress::Swapped) {}
    drop(c);
    let c = RedbCatalog::open(&path).unwrap();
    assert_eq!(c.invalidation_revision(raw).unwrap(), finished);
}

fn invalidation_revision_tracks_late_purge_effects<C: Catalog>(c: C) {
    let (raw, derived) = graph(&c);
    c.start_purge(PurgeSelector::Ids(vec![id(1)]), "test", "", id(603), 1).unwrap();
    while present(&c, 1) {
        c.purge_step(id(603), 1).unwrap();
    }
    let before = c.invalidation_revision(derived).unwrap().unwrap();
    c.insert(vec![new(header(81, derived, 90, 91, 0, &[1], &[]))]).unwrap();
    assert!(!present(&c, 81));
    assert!(c.invalidation_revision(derived).unwrap().unwrap() > before);
    assert!(c.invalidation_revision(raw).unwrap().unwrap() > 0);
}

fn streams_register_once_per_name<C: Catalog>(c: C) {
    let a = c.register_stream(info("cam0")).unwrap();
    assert_eq!(c.register_stream(info("cam0")).unwrap(), a, "the same info again is the same stream");
    let b = c.register_stream(info("cam1")).unwrap();
    assert_ne!(a, b);
    let mut other = info("cam0");
    other.kind = StreamKind::Derived;
    assert!(matches!(c.register_stream(other), Err(CatalogError::Conflict(_))));
    assert_eq!(c.stream_id("cam1").unwrap(), Some(b));
    assert_eq!(c.stream(a).unwrap(), Some(info("cam0")));
    assert_eq!(c.stream_id("nope").unwrap(), None);
}

fn a_time_scan_returns_exactly_the_window_in_order_across_seq_ties<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    let other = c.register_stream(info("cam1")).unwrap();
    // Instants at 10 (seq 0, 1, 2), 20, 30, and one on another stream at 20.
    let batch = vec![
        new(header(3, s, 10, 10, 2, &[], &[])),
        new(header(1, s, 10, 10, 0, &[], &[])),
        new(header(2, s, 10, 10, 1, &[], &[])),
        new(header(4, s, 20, 20, 0, &[], &[])),
        new(header(5, s, 30, 30, 0, &[], &[])),
        new(header(6, other, 20, 20, 0, &[], &[])),
    ];
    c.insert(batch).unwrap();
    assert_eq!(ids(&c.range(s, iv(10, 30), None, 100).unwrap()), vec![id(1), id(2), id(3), id(4)]);
    assert_eq!(ids(&c.range(s, iv(11, 20), None, 100).unwrap()), Vec::<Uuid>::new(), "half-open: 20 is out");
    assert_eq!(ids(&c.range(s, iv(30, 31), None, 100).unwrap()), vec![id(5)]);
}

fn a_range_pages_through_with_its_cursor<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    c.insert((0..5u16).map(|i| new(header(i + 1, s, 10, 10, u32::from(i), &[], &[]))).collect()).unwrap();
    let first = c.range(s, iv(0, 100), None, 2).unwrap();
    assert_eq!(ids(&first), vec![id(1), id(2)]);
    assert_eq!(first.next, Some(RangeCursor { start_ns: 10, seq: 1 }));
    let second = c.range(s, iv(0, 100), first.next, 2).unwrap();
    assert_eq!(ids(&second), vec![id(3), id(4)]);
    let last = c.range(s, iv(0, 100), second.next, 2).unwrap();
    assert_eq!((ids(&last), last.next), (vec![id(5)], None));
}

fn a_payload_starting_before_the_window_but_reaching_into_it_is_found<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    c.insert(vec![new(header(1, s, 0, 5_000, 0, &[], &[])), new(header(2, s, 100, 100, 0, &[], &[]))]).unwrap();
    assert_eq!(ids(&c.range(s, iv(4_000, 4_001), None, 10).unwrap()), vec![id(1)]);
    assert_eq!(ids(&c.range(s, iv(5_000, 6_000), None, 10).unwrap()), Vec::<Uuid>::new(), "it ends at 5000");
}

fn a_bad_batch_writes_nothing<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    c.insert(vec![new(header(1, s, 10, 10, 0, &[], &[]))]).unwrap();
    let unknown_parent = vec![new(header(2, s, 20, 20, 0, &[], &[])), new(header(3, s, 30, 30, 0, &[99], &[]))];
    assert!(matches!(c.insert(unknown_parent), Err(CatalogError::Invalid(_))));
    assert!(!present(&c, 2), "the good payload in the failed batch wasn't written");
    let same_key = vec![new(header(4, s, 10, 10, 0, &[], &[]))];
    assert!(matches!(c.insert(same_key), Err(CatalogError::Conflict(_))));
    assert!(matches!(c.insert(vec![new(header(1, s, 40, 40, 0, &[], &[]))]), Err(CatalogError::Conflict(_))));
    assert!(matches!(c.insert(vec![new(header(5, StreamId(999), 1, 1, 0, &[], &[]))]), Err(CatalogError::NotFound(_))));
    // A parent earlier in the same batch is fine.
    c.insert(vec![new(header(6, s, 50, 50, 0, &[], &[])), new(header(7, s, 60, 60, 0, &[6], &[]))]).unwrap();
    assert_eq!(c.parents(id(7)).unwrap(), vec![id(6)]);
}

fn locations_are_kept_per_tier<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    c.insert(vec![NewPayload { header: header(1, s, 10, 10, 0, &[], &[]), location: Some(shm(3)), coverage: None }])
        .unwrap();
    c.set_location(id(1), warm("file:///c/1")).unwrap();
    let locations = |c: &C| match c.get(id(1)).unwrap() {
        Some(Resolved::Payload(p)) => p.locations,
        other => panic!("{other:?}"),
    };
    assert_eq!(locations(&c), vec![shm(3), warm("file:///c/1")]);
    c.clear_location(id(1), StorageTier::Tier0Hot).unwrap();
    assert_eq!(locations(&c), vec![warm("file:///c/1")]);
    assert!(matches!(c.set_location(id(9), shm(1)), Err(CatalogError::NotFound(_))));
}

fn many_payloads_tier_copies_clear_at_once_and_the_unknown_are_skipped<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    for n in 1..=3u16 {
        let h = header(n, s, u64::from(n) * 10, u64::from(n) * 10 + 10, 0, &[], &[]);
        c.insert(vec![NewPayload { header: h, location: Some(shm(u32::from(n))), coverage: None }]).unwrap();
    }
    c.set_location(id(2), warm("file:///c/2")).unwrap();
    let locations = |c: &C, n: u16| match c.get(id(n)).unwrap() {
        Some(Resolved::Payload(p)) => p.locations,
        other => panic!("{other:?}"),
    };
    assert_eq!(c.clear_locations(&[id(1), id(2), id(9)], StorageTier::Tier0Hot).unwrap(), 2, "9 isn't indexed");
    assert_eq!(locations(&c, 1), vec![]);
    assert_eq!(locations(&c, 2), vec![warm("file:///c/2")], "only the tier asked for goes");
    assert_eq!(locations(&c, 3), vec![shm(3)], "a payload not named keeps its copy");
}

fn provenance_goes_both_ways_and_descendants_are_transitive<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    let d = c.register_stream(info("derived")).unwrap();
    // 1 -> 2 -> 4, 1 -> 3 -> 4
    c.insert(vec![
        new(header(1, s, 10, 10, 0, &[], &[])),
        new(header(2, d, 10, 10, 0, &[1], &[])),
        new(header(3, d, 10, 10, 1, &[1], &[])),
        new(header(4, d, 10, 10, 2, &[2, 3], &[])),
    ])
    .unwrap();
    assert_eq!(c.children(id(1)).unwrap(), vec![id(2), id(3)]);
    assert_eq!(c.parents(id(4)).unwrap(), vec![id(2), id(3)]);
    assert_eq!(c.descendants(&[id(1)], 10).unwrap(), vec![id(2), id(3), id(4)]);
    assert_eq!(c.descendants(&[id(1)], 2).unwrap(), vec![id(2), id(3)]);
    assert_eq!(c.descendants(&[id(4)], 10).unwrap(), Vec::<Uuid>::new());
}

/// Three frames on cam0; frame 2 has a crop derived from it. A chunk absorbs all three.
fn chunked<C: Catalog>(c: &C) -> StreamId {
    let s = c.register_stream(info("cam0")).unwrap();
    let d = c.register_stream(info("face.crop")).unwrap();
    c.insert(vec![
        NewPayload { header: header(1, s, 0, 0, 0, &[], &[]), location: Some(shm(1)), coverage: None },
        NewPayload { header: header(2, s, 33, 33, 0, &[], &[]), location: Some(shm(2)), coverage: None },
        NewPayload { header: header(3, s, 66, 66, 0, &[], &[]), location: Some(shm(3)), coverage: None },
        new(header(10, d, 33, 33, 0, &[2], &[])),
    ])
    .unwrap();
    let frames = [0, 33, 66].map(|t| ChunkFrame::new(t, t * 10, 300)).to_vec();
    let chunk = NewChunk {
        payload: NewPayload {
            header: header(100, s, 0, 99, 0, &[], &[]),
            location: Some(warm("file:///c/0")),
            coverage: None,
        },
        frames,
    };
    c.commit_chunk(chunk, &[id(1), id(2), id(3)]).unwrap();
    s
}

fn frame_provenance_rides_with_its_chunk_through_a_rewrite_until_the_chunk_goes<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    c.insert(vec![
        NewPayload { header: header(1, s, 0, 0, 0, &[], &[]), location: Some(shm(1)), coverage: None },
        NewPayload { header: header(2, s, 33, 33, 0, &[], &[]), location: Some(shm(2)), coverage: None },
    ])
    .unwrap();
    let frames = [0, 33].map(|t| ChunkFrame::new(t, t * 10, 300)).to_vec();
    let chunk = || NewChunk {
        payload: NewPayload {
            header: header(100, s, 0, 99, 0, &[], &[]),
            location: Some(warm("file:///c/0")),
            coverage: None,
        },
        frames: frames.clone(),
    };
    let provenance = ChunkProvenance::new(vec![
        FrameProvenance { device_seq: 7, epoch: 42, stamped_at_arrival: false },
        FrameProvenance { device_seq: 8, epoch: 42, stamped_at_arrival: true },
    ]);
    // One entry short of the frames: refused, and nothing lands.
    let short = ChunkProvenance::new(provenance.frames[..1].to_vec());
    let refused = c.commit_chunk_with_provenance(chunk(), &[id(1), id(2)], Some(&short));
    assert!(matches!(refused, Err(CatalogError::Invalid(_))), "{refused:?}");
    assert!(matches!(c.get(id(1)).unwrap(), Some(Resolved::Payload(_))), "the frames weren't absorbed");
    assert!(c.get(id(100)).unwrap().is_none());

    c.commit_chunk_with_provenance(chunk(), &[id(1), id(2)], Some(&provenance)).unwrap();
    assert_eq!(c.chunk_provenance(id(100)).unwrap(), Some(provenance.clone()));
    assert_eq!(c.chunk_snapshot(id(100)).unwrap().unwrap().provenance, Some(provenance.clone()));
    // A rewrite keeps the frames in place, so their provenance stays.
    c.rewrite_chunk(id(100), frames.clone(), warm("file:///c/1")).unwrap();
    assert_eq!(c.chunk_provenance(id(100)).unwrap(), Some(provenance));

    c.start_purge(PurgeSelector::Ids(vec![id(100)]), "test", "", id(600), 1).unwrap();
    while !matches!(c.purge_step(id(600), 1).unwrap(), PurgeProgress::Removed { .. }) {}
    assert_eq!(c.chunk_provenance(id(100)).unwrap(), None);
}

fn a_chunk_committed_without_provenance_has_none_rather_than_an_empty_claim<C: Catalog>(c: C) {
    chunked(&c);
    assert_eq!(c.chunk_provenance(id(100)).unwrap(), None);
    assert_eq!(c.chunk_snapshot(id(100)).unwrap().unwrap().provenance, None);
}

fn a_repeated_chunk_commit_is_a_duplicate_with_its_sequence_and_a_changed_one_is_refused<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    c.insert(vec![
        NewPayload { header: header(1, s, 0, 0, 0, &[], &[]), location: Some(shm(1)), coverage: None },
        NewPayload { header: header(2, s, 33, 33, 0, &[], &[]), location: Some(shm(2)), coverage: None },
    ])
    .unwrap();
    let chunk = NewChunk {
        payload: NewPayload {
            header: header(100, s, 0, 66, 0, &[], &[]),
            location: Some(warm("file:///c/0")),
            coverage: None,
        },
        frames: [0, 33].map(|t| ChunkFrame::new(t, t * 10, 300)).to_vec(),
    };
    let first = c.commit_chunk(chunk.clone(), &[id(1), id(2)]).unwrap();
    assert!(!first.duplicate && first.seq > 0);
    // The worker's retry after a lost reply: the frames are compacted by now, and the same bytes are a success.
    let again = c.commit_chunk(chunk, &[id(1), id(2)]).unwrap();
    assert_eq!((again.seq, again.duplicate, again.hidden), (first.seq, true, false));
    // A retry with the same header but a bad shape is refused on both catalogs alike (the shape checks come first).
    let mut bad = NewChunk {
        payload: NewPayload {
            header: header(100, s, 0, 66, 0, &[], &[]),
            location: Some(warm("file:///c/0")),
            coverage: None,
        },
        frames: [0, 33].map(|t| ChunkFrame::new(t, t * 10, 300)).to_vec(),
    };
    bad.frames.pop();
    assert!(matches!(c.commit_chunk(bad, &[id(1), id(2)]), Err(CatalogError::Invalid(_))));
    // The same id with a different header is not a repeat.
    let changed = NewChunk {
        payload: NewPayload {
            header: header(100, s, 0, 99, 0, &[], &[]),
            location: Some(warm("file:///c/0")),
            coverage: None,
        },
        frames: [0, 33].map(|t| ChunkFrame::new(t, t * 10, 300)).to_vec(),
    };
    assert!(c.commit_chunk(changed, &[id(1), id(2)]).is_err());
}

fn a_retried_chunk_commit_under_a_running_purge_is_a_hidden_duplicate<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    c.insert(vec![
        NewPayload { header: header(1, s, 0, 0, 0, &[], &[]), location: Some(shm(1)), coverage: None },
        NewPayload { header: header(2, s, 33, 33, 0, &[], &[]), location: Some(shm(2)), coverage: None },
    ])
    .unwrap();
    let chunk = NewChunk {
        payload: NewPayload {
            header: header(100, s, 0, 66, 0, &[], &[]),
            location: Some(warm("file:///c/0")),
            coverage: None,
        },
        frames: [0, 33].map(|t| ChunkFrame::new(t, t * 10, 300)).to_vec(),
    };
    let first = c.commit_chunk(chunk.clone(), &[id(1), id(2)]).unwrap();
    // A purge dooms the chunk (its first step hides the roots) but hasn't removed it yet.
    c.start_purge(PurgeSelector::Range { span: iv(0, 100), streams: Some(vec![s]) }, "t", "", id(900), 1).unwrap();
    assert!(matches!(c.purge_step(id(900), 1).unwrap(), PurgeProgress::Running { .. }));
    // The tier worker's retry of the same commit (its reply was lost) is a duplicate that the purge hides: the
    // broker adds no copy for it.
    let again = c.commit_chunk(chunk, &[id(1), id(2)]).unwrap();
    assert_eq!((again.seq, again.duplicate, again.hidden), (first.seq, true, true));
}

fn a_chunk_commit_meeting_a_purge_marks_the_purged_frames_cut_and_absorbs_the_rest<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    let d = c.register_stream(info("face.crop")).unwrap();
    c.insert(vec![
        NewPayload { header: header(1, s, 0, 0, 0, &[], &[]), location: Some(shm(1)), coverage: None },
        NewPayload { header: header(2, s, 33, 33, 0, &[], &[]), location: Some(shm(2)), coverage: None },
        NewPayload { header: header(3, s, 66, 66, 0, &[], &[]), location: Some(shm(3)), coverage: None },
        NewPayload { header: header(4, s, 99, 99, 0, &[], &[]), location: Some(shm(4)), coverage: None },
        new(header(10, d, 66, 66, 0, &[3], &[])),
    ])
    .unwrap();
    // A purge over frame 2 has run to the end (its row is gone); another over frame 4 has taken its first step
    // (doomed, not removed yet). The chunk the tier worker closes meanwhile absorbs all four.
    c.purge(PurgeSelector::Range { span: iv(33, 34), streams: Some(vec![s]) }, "t", "", id(501), 1).unwrap();
    c.start_purge(PurgeSelector::Range { span: iv(99, 100), streams: Some(vec![s]) }, "t", "", id(502), 1).unwrap();
    assert!(matches!(c.purge_step(id(502), 1).unwrap(), PurgeProgress::Running { .. }));
    let chunk = NewChunk {
        payload: NewPayload {
            header: header(100, s, 0, 132, 0, &[], &[]),
            location: Some(warm("file:///c/0")),
            coverage: None,
        },
        frames: [0, 33, 66, 99].map(|t| ChunkFrame::new(t, t * 10, 300)).to_vec(),
    };
    let committed = c.commit_chunk(chunk, &[id(1), id(2), id(3), id(4)]).unwrap();
    assert!(!committed.duplicate && !committed.hidden);
    // Frames 1 and 3 are absorbed (3 keeps a row for its child); the chunk lands without 2 (gone, which nothing will
    // report again) and 4 (hidden: the purge that hid it reports the cut).
    assert_eq!(committed.cut, vec![1]);
    assert!(present(&c, 1), "absorbed: its row maps it to the chunk, so a late child still finds its parent");
    // The doomed frame's row is the purge's to remove, not the commit's: the second purge cuts the slot it hid the
    // frame of (a step reports it, as for a chunk found in place), finishes, and reports the frame removed; the
    // chunk stays, with both frames cut.
    let (deltas, report) = step_through(&c, id(502));
    assert!(deltas.iter().any(|d| d.cut_frames.iter().any(|f| f.chunk == id(100) && f.index == 3)), "{deltas:?}");
    assert!(report.removed.iter().any(|r| r.id == id(4)), "{report:?}");
    assert!(report.partial_chunks.iter().any(|p| p.chunk == id(100) && p.frames == [3]), "{report:?}");
    assert!(!present(&c, 4));
    assert!(matches!(c.get(id(100)).unwrap(), Some(Resolved::Payload(_))));
    match c.get(id(3)).unwrap() {
        Some(Resolved::AbsorbedFrame { chunk, frame_index }) => {
            assert_eq!((chunk.header.id(), frame_index), (id(100), 2));
            let frames = chunk.frames.unwrap();
            assert_eq!(frames.iter().map(|f| f.cut).collect::<Vec<_>>(), [false, true, false, true]);
        }
        other => panic!("{other:?}"),
    }
    // A live frame the worker marks cut (a purge reached the file first) isn't absorbed: it stays its own payload,
    // the chunk lands without its slot, and the chunk takes the frame's time key (the same start, seq 0).
    let s2 = c.register_stream(info("cam1")).unwrap();
    c.insert(vec![NewPayload { header: header(5, s2, 0, 0, 0, &[], &[]), location: Some(shm(5)), coverage: None }])
        .unwrap();
    let mut cut = ChunkFrame::new(0, 0, 300);
    cut.cut = true;
    let chunk = NewChunk {
        payload: NewPayload {
            header: header(101, s2, 0, 1, 0, &[], &[]),
            location: Some(warm("file:///c/1")),
            coverage: None,
        },
        frames: vec![cut],
    };
    let committed = c.commit_chunk(chunk, &[id(5)]).unwrap();
    assert_eq!((committed.hidden, committed.cut), (false, vec![0]));
    assert!(matches!(c.get(id(5)).unwrap(), Some(Resolved::Payload(_))), "the worker's cut leaves the frame as it was");
    assert_eq!(ids(&c.range(s2, iv(0, 1), None, 10).unwrap()), vec![id(101)], "the chunk holds the instant's key");
}

fn a_chunk_whose_first_frame_a_purge_hid_commits_under_that_frames_key<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    c.insert(
        (1..=3)
            .map(|n| NewPayload {
                header: header(n, s, u64::from(n - 1) * 33, u64::from(n - 1) * 33, 0, &[], &[]),
                location: Some(shm(u32::from(n))),
                coverage: None,
            })
            .collect(),
    )
    .unwrap();
    // A purge over the first frame has hidden it when the chunk commits. The chunk's own key (the stream, its start,
    // seq 0) is that frame's, as the tier worker builds chunk headers; the frame's row stays for the purge to remove.
    c.start_purge(PurgeSelector::Range { span: iv(0, 1), streams: Some(vec![s]) }, "t", "", id(530), 1).unwrap();
    assert!(matches!(c.purge_step(id(530), 1).unwrap(), PurgeProgress::Running { .. }));
    let chunk = NewChunk {
        payload: NewPayload {
            header: header(100, s, 0, 99, 0, &[], &[]),
            location: Some(warm("file:///c/0")),
            coverage: None,
        },
        frames: [0, 33, 66].map(|t| ChunkFrame::new(t, t * 10, 300)).to_vec(),
    };
    let committed = c.commit_chunk(chunk, &[id(1), id(2), id(3)]).unwrap();
    assert_eq!((committed.hidden, committed.cut), (false, vec![]), "the purge reports the cut, so the commit doesn't");
    assert_eq!(ids(&c.range(s, iv(0, 1), None, 10).unwrap()), vec![id(100)], "the chunk answers for the instant");
    let (_, report) = step_through(&c, id(530));
    assert!(report.removed.iter().any(|r| r.id == id(1)), "{report:?}");
    assert!(report.partial_chunks.iter().any(|p| p.chunk == id(100) && p.frames == [0]), "{report:?}");
    assert!(!present(&c, 1) && present(&c, 2) && present(&c, 100), "frame 2 stays absorbed, its row kept");
    assert_eq!(ids(&c.range(s, iv(0, 1), None, 10).unwrap()), vec![id(100)], "removing the frame left the chunk's key");
}

fn a_chunk_landing_across_a_purge_window_between_its_frames_has_that_slot_cut<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    c.insert(
        [0u64, 33, 45, 66]
            .iter()
            .zip(1u16..)
            .map(|(&t, n)| NewPayload {
                header: header(n, s, t, t, 0, &[], &[]),
                location: Some(shm(u32::from(n))),
                coverage: None,
            })
            .collect(),
    )
    .unwrap();
    // The window [40, 50) holds frame 3 (at 45), hidden at the first step, and reaches into frame 2's slot in the
    // chunk, [33, 45), though not frame 2's own instant: the chunk lands without both slots, by the rule a chunk
    // found in place follows, and the purge reports both cuts at its next step.
    c.start_purge(PurgeSelector::Range { span: iv(40, 50), streams: Some(vec![s]) }, "t", "", id(531), 1).unwrap();
    assert!(matches!(c.purge_step(id(531), 1).unwrap(), PurgeProgress::Running { .. }));
    let chunk = NewChunk {
        payload: NewPayload {
            header: header(100, s, 0, 99, 0, &[], &[]),
            location: Some(warm("file:///c/0")),
            coverage: None,
        },
        frames: [0, 33, 45, 66].map(|t| ChunkFrame::new(t, t * 10, 300)).to_vec(),
    };
    let committed = c.commit_chunk(chunk, &[id(1), id(2), id(3), id(4)]).unwrap();
    assert_eq!(
        (committed.hidden, committed.cut),
        (false, vec![]),
        "the purge reports both cuts, so the commit doesn't"
    );
    let (deltas, report) = step_through(&c, id(531));
    let mut cuts: Vec<(Uuid, u32)> =
        deltas.iter().flat_map(|d| d.cut_frames.iter().map(|f| (f.chunk, f.index))).collect();
    cuts.sort();
    assert_eq!(cuts, vec![(id(100), 1), (id(100), 2)]);
    assert!(report.partial_chunks.iter().any(|p| p.chunk == id(100) && p.frames == [1, 2]), "{report:?}");
    let page = c.range(s, iv(0, 100), None, 10).unwrap();
    let item = page.items.iter().find(|i| i.header.id() == id(100)).expect("the chunk stays");
    let flags: Vec<bool> = item.frames.as_ref().unwrap().iter().map(|f| f.cut).collect();
    assert_eq!(flags, [false, true, true, false]);
}

/// The chunk [`chunked`] committed (frames at 0, 33, and 66), grown by a part: frames 4 and 5, at 99 and 132.
fn part<C: Catalog>(c: &C, s: StreamId) -> NewChunk {
    c.insert(vec![
        NewPayload { header: header(4, s, 99, 99, 0, &[], &[]), location: Some(shm(4)), coverage: None },
        NewPayload { header: header(5, s, 132, 132, 0, &[], &[]), location: Some(shm(5)), coverage: None },
    ])
    .unwrap();
    NewChunk {
        payload: NewPayload {
            header: header(100, s, 0, 165, 0, &[], &[]),
            location: Some(StorageLocator::Chunk {
                tier: StorageTier::Tier1Warm,
                uri: "file:///c/0".into(),
                offset: 0,
                length: 1700,
            }),
            coverage: None,
        },
        frames: [99, 132].map(|t| ChunkFrame::new(t, t * 10, 300)).to_vec(),
    }
}

fn a_part_joins_its_chunk_absorbs_its_frames_and_moves_it_in_commit_order<C: Catalog>(c: C) {
    let s = chunked(&c);
    let Some(Resolved::Payload(before)) = c.get(id(100)).unwrap() else { panic!("the chunk") };
    let part = part(&c, s);
    let done = c.extend_chunk(part.clone(), 3, &[id(4), id(5)], None).unwrap();
    assert!(!done.duplicate && !done.hidden && done.cut.is_empty());
    assert!(done.seq > before.commit_seq, "a reader following commit order sees the chunk again");
    let Some(Resolved::Payload(after)) = c.get(id(100)).unwrap() else { panic!("the chunk") };
    assert_eq!(after.commit_seq, done.seq);
    assert_eq!((after.header.start_ns(), after.header.end_ns()), (0, 165));
    assert_eq!(after.locations, vec![part.payload.location.clone().unwrap()], "one Tier 1 copy, at its new length");
    assert_eq!(
        after.frames.unwrap().iter().map(|f| (f.start_ns, f.offset)).collect::<Vec<_>>(),
        [(0, 0), (33, 330), (66, 660), (99, 990), (132, 1320)]
    );
    match c.get(id(5)).unwrap() {
        Some(Resolved::AbsorbedFrame { chunk, frame_index }) => {
            assert_eq!((chunk.header.id(), frame_index), (id(100), 4), "its index in the whole chunk");
        }
        other => panic!("frame 5 should resolve through FRAMES, got {other:?}"),
    }
    assert_eq!(ids(&c.range(s, iv(0, 200), None, 10).unwrap()), vec![id(100)], "one row for the file");
    assert_eq!(ids(&c.range(s, iv(140, 150), None, 10).unwrap()), vec![id(100)], "found past its old end");
    // The same part again, after a lost reply.
    let again = c.extend_chunk(part, 3, &[id(4), id(5)], None).unwrap();
    assert_eq!((again.seq, again.duplicate), (done.seq, true));
    // A forget that cuts into the file cuts frames by their place in the whole chunk.
    c.purge(PurgeSelector::Range { span: iv(132, 133), streams: Some(vec![s]) }, "privacy", "", id(700), 50).unwrap();
    let Some(Resolved::Payload(cut)) = c.get(id(100)).unwrap() else { panic!("the chunk stays") };
    assert_eq!(cut_indexes(&cut.frames.unwrap()), vec![4]);
}

fn a_part_that_isnt_its_chunks_next_is_refused_and_changes_nothing<C: Catalog>(c: C) {
    let s = chunked(&c);
    let part = part(&c, s);
    let ids = [id(4), id(5)];
    let invalid = |r: Result<Inserted>| assert!(matches!(r, Err(CatalogError::Invalid(_))), "{r:?}");
    invalid(c.extend_chunk(part.clone(), 2, &ids, None)); // the chunk holds three frames
    invalid(c.extend_chunk(part.clone(), 0, &ids, None)); // a part never starts a chunk
    invalid(c.extend_chunk(part.clone(), 3, &ids[..1], None));
    invalid(c.extend_chunk(NewChunk { frames: Vec::new(), ..part.clone() }, 3, &[], None));
    invalid(c.extend_chunk(part.clone(), u32::try_from(MAX_CHUNK_FRAMES).unwrap() - 1, &ids, None));
    let with = |h: HeaderBuf| NewChunk { payload: NewPayload { header: h, ..part.payload.clone() }, ..part.clone() };
    invalid(c.extend_chunk(with(header(100, s, 1, 165, 0, &[], &[])), 3, &ids, None)); // another start
    invalid(c.extend_chunk(with(header(100, s, 0, 98, 0, &[], &[])), 3, &ids, None)); // ends before it did
    let mut backwards = part.clone();
    backwards.frames[0].start_ns = 50; // before the chunk's last frame
    invalid(c.extend_chunk(backwards, 3, &ids, None));
    let unknown = c.extend_chunk(with(header(101, s, 0, 165, 0, &[], &[])), 3, &ids, None);
    assert!(matches!(unknown, Err(CatalogError::NotFound(_))), "{unknown:?}");
    let Some(Resolved::Payload(chunk)) = c.get(id(100)).unwrap() else { panic!("the chunk") };
    assert_eq!((chunk.header.end_ns(), chunk.frames.unwrap().len()), (99, 3));
    assert!(matches!(c.get(id(4)).unwrap(), Some(Resolved::Payload(_))), "the part's frames keep their rows");
    // A purge that hid the whole chunk: the part's frames need another chunk.
    c.start_purge(PurgeSelector::Range { span: iv(0, 99), streams: Some(vec![s]) }, "t", "", id(900), 1).unwrap();
    assert!(matches!(c.purge_step(id(900), 1).unwrap(), PurgeProgress::Running { .. }));
    let hidden = c.extend_chunk(part, 3, &ids, None);
    assert!(matches!(hidden, Err(CatalogError::Conflict(_))), "{hidden:?}");
}

fn a_part_landing_under_a_purge_has_the_purged_frames_cut<C: Catalog>(c: C) {
    let s = chunked(&c);
    let part = part(&c, s);
    // Frame 5 is purged before its part commits: the part lands with that frame cut, and absorbs frame 4.
    c.purge(PurgeSelector::Ids(vec![id(5)]), "t", "", id(501), 50).unwrap();
    let done = c.extend_chunk(part, 3, &[id(4), id(5)], None).unwrap();
    assert_eq!(done.cut, vec![4], "the cut frame's index in the whole chunk");
    let Some(Resolved::Payload(chunk)) = c.get(id(100)).unwrap() else { panic!("the chunk") };
    assert_eq!(cut_indexes(&chunk.frames.unwrap()), vec![4]);
    assert!(matches!(c.get(id(4)).unwrap(), Some(Resolved::AbsorbedFrame { frame_index: 3, .. })));
    assert!(c.get(id(5)).unwrap().is_none());
}

fn a_parts_provenance_joins_the_chunks_and_a_silent_side_reads_as_unknown<C: Catalog>(c: C) {
    let attested = |seq| FrameProvenance { device_seq: seq, epoch: 42, stamped_at_arrival: false };
    let s = chunked(&c);
    let tail = ChunkProvenance::new(vec![attested(10), attested(11)]);
    c.extend_chunk(part(&c, s), 3, &[id(4), id(5)], Some(&tail)).unwrap();
    let unknown = FrameProvenance::default();
    assert_eq!(
        c.chunk_provenance(id(100)).unwrap().unwrap().frames,
        [unknown, unknown, unknown, attested(10), attested(11)],
        "the chunk's first part attested nothing"
    );
    c.insert(vec![NewPayload { header: header(6, s, 165, 165, 0, &[], &[]), location: Some(shm(6)), coverage: None }])
        .unwrap();
    let last = NewChunk {
        payload: NewPayload {
            header: header(100, s, 0, 198, 0, &[], &[]),
            location: Some(warm("file:///c/0")),
            coverage: None,
        },
        frames: vec![ChunkFrame::new(165, 1650, 300)],
    };
    let short = ChunkProvenance::new(Vec::new());
    assert!(matches!(c.extend_chunk(last.clone(), 5, &[id(6)], Some(&short)), Err(CatalogError::Invalid(_))));
    c.extend_chunk(last, 5, &[id(6)], None).unwrap();
    assert_eq!(c.chunk_provenance(id(100)).unwrap().unwrap().frames[3..], [attested(10), attested(11), unknown]);
}

fn a_chunk_commit_keeps_every_absorbed_frame_resolvable<C: Catalog>(c: C) {
    let s = chunked(&c);
    assert!(present(&c, 1) && present(&c, 3), "frames nothing came from yet keep their rows");
    match c.get(id(1)).unwrap() {
        Some(Resolved::AbsorbedFrame { chunk, frame_index }) => {
            assert_eq!((chunk.header.id(), frame_index), (id(100), 0))
        }
        other => panic!("frame 1 should resolve through FRAMES, got {other:?}"),
    }
    match c.get(id(2)).unwrap() {
        Some(Resolved::AbsorbedFrame { chunk, frame_index }) => {
            assert_eq!((chunk.header.id(), frame_index), (id(100), 1));
            assert_eq!(chunk.frames.unwrap()[1], ChunkFrame::new(33, 330, 300));
        }
        other => panic!("frame 2 should resolve through FRAMES, got {other:?}"),
    }
    assert_eq!(c.parents(id(10)).unwrap(), vec![id(2)], "the crop's parent still resolves");
    assert_eq!(ids(&c.range(s, iv(0, 200), None, 10).unwrap()), vec![id(100)], "the stream holds the chunk now");
}

fn an_identical_header_for_an_absorbed_frame_is_a_repeat_answering_the_chunks_sequence<C: Catalog>(c: C) {
    // The tier worker registering audio in place twice: the batches' headers again, after the chunk absorbed them.
    let s = chunked(&c);
    let chunk_seq = match c.get(id(100)).unwrap() {
        Some(Resolved::Payload(p)) => p.commit_seq,
        other => panic!("{other:?}"),
    };
    let again = c
        .insert(vec![
            NewPayload { header: header(1, s, 0, 0, 0, &[], &[]), location: Some(shm(1)), coverage: None },
            NewPayload { header: header(3, s, 66, 66, 0, &[], &[]), location: None, coverage: None },
        ])
        .unwrap();
    assert_eq!(
        again.iter().map(|i| (i.duplicate, i.seq, i.hidden)).collect::<Vec<_>>(),
        vec![(true, chunk_seq, false); 2]
    );
    assert!(matches!(c.get(id(1)).unwrap(), Some(Resolved::AbsorbedFrame { .. })), "nothing new was written");
    assert_eq!(
        ids(&c.range(s, iv(0, 100), None, 10).unwrap()),
        vec![id(100)],
        "the stream still holds the chunk alone"
    );
    // The same id at another start is no repeat of the absorbed frame: a conflict, as any reused id is.
    assert!(matches!(
        c.insert(vec![NewPayload { header: header(1, s, 5, 5, 0, &[], &[]), location: None, coverage: None }]),
        Err(CatalogError::Conflict(_))
    ));
    // Nor is a header on another stream.
    let other = c.register_stream(info("cam1")).unwrap();
    assert!(matches!(
        c.insert(vec![NewPayload { header: header(1, other, 0, 0, 0, &[], &[]), location: None, coverage: None }]),
        Err(CatalogError::Conflict(_))
    ));
}

fn a_windowed_sequence_query_reads_the_window_not_the_history<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    // History: 300 payloads long before the window, committed first; then an import commits a late one (an old
    // start, committed last) into the window; a long chunk starting before the window reaches into it.
    let old: Vec<NewPayload> =
        (1..=300).map(|n| new(header(n, s, u64::from(n) * 10, u64::from(n) * 10, 0, &[], &[]))).collect();
    c.insert(old).unwrap();
    c.insert(vec![new(header(1001, s, 10_000, 10_000, 0, &[], &[900]))]).unwrap();
    c.insert(vec![new(header(1002, s, 10_010, 10_010, 0, &[], &[]))]).unwrap();
    c.insert(vec![new(header(1003, s, 9_000, 10_005, 1, &[], &[]))]).unwrap(); // starts before, reaches in
    c.insert(vec![new(header(1004, s, 10_020, 10_020, 0, &[], &[]))]).unwrap();
    c.insert(vec![new(header(1005, s, 20_000, 20_000, 0, &[], &[]))]).unwrap(); // after the window
    let window = QueryFilter { window: Some(iv(10_000, 10_100)), ..QueryFilter::default() };
    let ids = |p: &SeqPage| p.items.iter().map(|i| i.header.id()).collect::<Vec<_>>();
    // A scan cap of 1 would stop a history scan at its first row; the windowed query reads the window instead.
    let page = c.query(s, 0, &window, 100, 1).unwrap();
    assert_eq!(ids(&page), vec![id(1001), id(1002), id(1003), id(1004)], "in commit order, the run-up's chunk too");
    assert_eq!(page.next_after, None);
    // Paged by commit sequence, as the cursor always was.
    let first = c.query(s, 0, &window, 2, 1).unwrap();
    assert_eq!(ids(&first), vec![id(1001), id(1002)]);
    let after = first.next_after.expect("more to come");
    let rest = c.query(s, after, &window, 2, 1).unwrap();
    assert_eq!((ids(&rest), rest.next_after), (vec![id(1003), id(1004)], None));
    // The other filters still apply.
    let theirs = QueryFilter { subjects: Some(vec![id(900)]), ..window.clone() };
    assert_eq!(ids(&c.query(s, 0, &theirs, 100, 1).unwrap()), vec![id(1001)]);
    let empty = QueryFilter { window: Some(iv(5_000, 5_005)), ..QueryFilter::default() };
    assert_eq!(c.query(s, 0, &empty, 100, 1).unwrap().items, Vec::new());
}

fn a_child_committed_after_its_parent_was_absorbed_is_kept_and_purged_with_it<C: Catalog>(c: C) {
    // An utterance names the PCM batches it covers; ASR commits it seconds after the tier worker absorbed them.
    let s = chunked(&c);
    let d = c.register_stream(info("speech.utterance")).unwrap();
    c.insert(vec![new(header(11, d, 0, 70, 0, &[1, 3], &[]))]).expect("the absorbed frames still resolve as parents");
    assert_eq!(c.parents(id(11)).unwrap(), vec![id(1), id(3)]);
    assert!(c.descendants(&[id(1)], 10).unwrap().contains(&id(11)), "lineage runs through the absorbed frame");
    // Purging frame 1's slot takes the child with it, as it would have had the frame still been its own payload.
    let report = c.purge(PurgeSelector::Range { span: iv(0, 1), streams: Some(vec![s]) }, "t", "", id(505), 1).unwrap();
    assert_eq!(report.removed_frames, vec![id(1)]);
    assert_eq!(report.removed.iter().map(|r| r.id).collect::<Vec<_>>(), vec![id(11)]);
    assert!(!present(&c, 1) && !present(&c, 11) && present(&c, 100));
}

fn a_chunk_that_doesnt_match_its_frames_is_refused<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    c.insert(vec![new(header(1, s, 0, 0, 0, &[], &[]))]).unwrap();
    let chunk = |loc| NewChunk {
        payload: NewPayload { header: header(100, s, 0, 99, 1, &[], &[]), location: Some(loc), coverage: None },
        frames: vec![ChunkFrame::new(0, 0, 1)],
    };
    assert!(matches!(c.commit_chunk(chunk(warm("x")), &[]), Err(CatalogError::Invalid(_))));
    assert!(matches!(c.commit_chunk(chunk(warm("x")), &[id(9)]), Err(CatalogError::Invalid(_))));
    assert!(matches!(c.commit_chunk(chunk(shm(1)), &[id(1)]), Err(CatalogError::Invalid(_))));
    assert!(present(&c, 1) && !present(&c, 100));
}

fn coverage_merges_and_gaps_fill_the_rest<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    let stage = StageId(7);
    c.add_coverage(CoverageMark { source: s, stage, span: iv(10, 20) }).unwrap();
    c.add_coverage(CoverageMark { source: s, stage, span: iv(20, 30) }).unwrap();
    c.add_coverage(CoverageMark { source: s, stage, span: iv(50, 60) }).unwrap();
    assert_eq!(c.coverage(s, stage, iv(0, 100)).unwrap(), vec![iv(10, 30), iv(50, 60)]);
    assert_eq!(c.coverage_gaps(s, stage, iv(0, 100), 64).unwrap(), vec![iv(0, 10), iv(30, 50), iv(60, 100)]);
    assert_eq!(c.coverage_gaps(s, StageId(8), iv(0, 100), 64).unwrap(), vec![iv(0, 100)], "another stage did none");
    assert!(matches!(
        c.add_coverage(CoverageMark { source: StreamId(999), stage, span: iv(1, 2) }),
        Err(CatalogError::NotFound(_))
    ));
}

/// frames 1..=4 on cam0 at 10, 20, 30, 40; crops 11 (from 1, subject 900) and 12 (from 2); embeddings 21 (from 11)
/// and 22 (from 12); a summary 31 from both embeddings.
fn graph<C: Catalog>(c: &C) -> (StreamId, StreamId) {
    let cam = c.register_stream(info("cam0")).unwrap();
    let d = c.register_stream(info("derived")).unwrap();
    c.insert(vec![
        new(header(1, cam, 10, 10, 0, &[], &[])),
        new(header(2, cam, 20, 20, 0, &[], &[])),
        new(header(3, cam, 30, 30, 0, &[], &[])),
        new(header(4, cam, 40, 40, 0, &[], &[])),
        new(header(11, d, 10, 10, 0, &[1], &[900])),
        new(header(12, d, 20, 20, 0, &[2], &[])),
        new(header(21, d, 10, 10, 1, &[11], &[900])),
        new(header(22, d, 20, 20, 1, &[12], &[])),
        new(header(31, d, 10, 20, 2, &[21, 22], &[])),
    ])
    .unwrap();
    (cam, d)
}

fn a_range_purge_removes_exactly_the_descendants<C: Catalog>(c: C) {
    let (cam, _) = graph(&c);
    c.forget(ForgottenHash([5; 32])).unwrap();
    let report =
        c.purge(PurgeSelector::Range { span: iv(10, 11), streams: Some(vec![cam]) }, "test", "", id(500), 77).unwrap();
    let mut gone: Vec<Uuid> = report.removed.iter().map(|r| r.id).collect();
    gone.sort();
    assert_eq!(gone, vec![id(1), id(11), id(21), id(31)]);
    let report = report.clone();
    for n in [2, 3, 4, 12, 22] {
        assert!(present(&c, n), "{n} wasn't derived from frame 1");
    }
    assert_eq!(c.children(id(22)).unwrap(), Vec::<Uuid>::new(), "no edge left to the purged summary");
    assert!(c.is_forgotten(ForgottenHash([5; 32])).unwrap(), "FORGOTTEN is never purged");
    let record = c.purge_record(id(500)).unwrap().unwrap();
    assert_eq!(
        (record.at_ns, record.reason.as_str(), record.report, record.completed_ns),
        (77, "test", Some(report), None)
    );
    assert_eq!(c.pending_purges().unwrap(), vec![id(500)]);
    c.complete_purge(id(500), 90).unwrap();
    assert_eq!(c.purge_record(id(500)).unwrap().unwrap().completed_ns, Some(90));
    assert_eq!(c.pending_purges().unwrap(), Vec::<Uuid>::new());
    assert!(matches!(c.purge(PurgeSelector::Entity(id(1)), "again", "", id(500), 78), Err(CatalogError::Conflict(_))));
}

fn an_entity_purge_removes_its_subject_and_everything_after<C: Catalog>(c: C) {
    graph(&c);
    let report = c.purge(PurgeSelector::Entity(id(900)), "forget", "", id(501), 1).unwrap();
    let mut gone: Vec<Uuid> = report.removed.iter().map(|r| r.id).collect();
    gone.sort();
    assert_eq!(gone, vec![id(11), id(21), id(31)]);
    assert!(present(&c, 1), "the frame isn't about the subject; only what was derived as being about them goes");
    assert_eq!(c.children(id(1)).unwrap(), Vec::<Uuid>::new());
}

fn a_purge_takes_the_purged_spans_out_of_coverage<C: Catalog>(c: C) {
    let (cam, _) = graph(&c);
    c.add_coverage(CoverageMark { source: cam, stage: StageId(1), span: iv(0, 100) }).unwrap();
    c.purge(PurgeSelector::Range { span: iv(10, 21), streams: Some(vec![cam]) }, "t", "", id(502), 1).unwrap();
    assert_eq!(c.coverage(cam, StageId(1), iv(0, 100)).unwrap(), vec![iv(0, 10), iv(21, 100)]);
}

fn a_range_purge_inside_a_chunk_cuts_frames_and_their_descendants<C: Catalog>(c: C) {
    let s = chunked(&c);
    let report =
        c.purge(PurgeSelector::Range { span: iv(35, 40), streams: Some(vec![s]) }, "t", "", id(503), 1).unwrap();
    assert_eq!(report.partial_chunks.len(), 1);
    assert_eq!((report.partial_chunks[0].chunk, report.partial_chunks[0].frames.clone()), (id(100), vec![1]));
    assert_eq!(report.removed_frames, vec![id(2)]);
    assert_eq!(report.removed.iter().map(|r| r.id).collect::<Vec<_>>(), vec![id(10)], "the crop from frame 2");
    assert!(present(&c, 100) && !present(&c, 2));
}

fn purging_a_whole_chunk_takes_its_absorbed_frames_too<C: Catalog>(c: C) {
    let s = chunked(&c);
    let report =
        c.purge(PurgeSelector::Range { span: iv(0, 100), streams: Some(vec![s]) }, "t", "", id(504), 1).unwrap();
    let mut gone: Vec<Uuid> = report.removed.iter().map(|r| r.id).collect();
    gone.sort();
    assert_eq!(gone, vec![id(10), id(100)]);
    assert_eq!(report.removed[0].locations.len() + report.removed[1].locations.len(), 1, "the chunk's file");
    let mut frames = report.removed_frames.clone();
    frames.sort();
    assert_eq!((frames, report.partial_chunks), (vec![id(1), id(2), id(3)], vec![]), "every absorbed frame's row");
}

fn an_entity_purge_of_an_absorbed_frame_takes_its_hydrated_copies_and_coverage<C: Catalog>(c: C) {
    // Frame 2 (subject 900) is absorbed into chunk 100, then hydrated back as 40 (parent: the chunk), and 42 is
    // derived from that copy. Forgetting 900 must take 40 and 42 too, and the frame's time out of coverage.
    let s = c.register_stream(info("cam0")).unwrap();
    let d = c.register_stream(info("derived")).unwrap();
    c.insert(vec![
        new(header(1, s, 0, 0, 0, &[], &[])),
        new(header(2, s, 33, 33, 0, &[], &[900])),
        new(header(3, s, 66, 66, 0, &[], &[])),
    ])
    .unwrap();
    let frames = [0, 33, 66].map(|t| ChunkFrame::new(t, t * 10, 300)).to_vec();
    let chunk = NewChunk {
        payload: NewPayload {
            header: header(100, s, 0, 99, 0, &[], &[]),
            location: Some(warm("file:///c/0")),
            coverage: None,
        },
        frames,
    };
    c.commit_chunk(chunk, &[id(1), id(2), id(3)]).unwrap();
    c.add_coverage(CoverageMark { source: s, stage: StageId(1), span: iv(0, 100) }).unwrap();
    c.insert(vec![
        new(header(40, s, 33, 33, 1, &[100], &[])),
        new(header(41, s, 66, 66, 1, &[100], &[])),
        new(header(42, d, 33, 33, 0, &[40], &[])),
    ])
    .unwrap();
    let report = c.purge(PurgeSelector::Entity(id(900)), "forget", "", id(505), 1).unwrap();
    let mut gone: Vec<Uuid> = report.removed.iter().map(|r| r.id).collect();
    gone.sort();
    assert_eq!(gone, vec![id(40), id(42)], "the hydrated copy of frame 2 and what came from it");
    assert_eq!(report.removed_frames, vec![id(2)]);
    assert_eq!(
        report.partial_chunks.iter().map(|p| (p.chunk, p.frames.clone())).collect::<Vec<_>>(),
        vec![(id(100), vec![1])]
    );
    assert!(present(&c, 41) && present(&c, 100), "frame 3's copy and the chunk stay");
    assert_eq!(c.coverage(s, StageId(1), iv(0, 100)).unwrap(), vec![iv(0, 33), iv(66, 100)], "frame 2's slot");
}

fn a_sensor_stream_re_registers_after_a_restart_with_a_fresh_anchor<C: Catalog>(c: C) {
    let mut first = info("cam0");
    first.anchor = Some(ClockAnchor { boot: BootId([1; 16]), mono_ns: 5, utc_ns: 50 });
    let s = c.register_stream(first).unwrap();
    let mut again = info("cam0");
    again.anchor = Some(ClockAnchor { boot: BootId([2; 16]), mono_ns: 7, utc_ns: 70 });
    again.rate_hz = Some(15.0);
    assert_eq!(c.register_stream(again.clone()).unwrap(), s);
    assert_eq!(c.stream(s).unwrap(), Some(again), "the new anchor and rate are kept");
    let mut other = info("cam0");
    other.shape_name = "MJPEG".into();
    assert!(matches!(c.register_stream(other), Err(CatalogError::Conflict(_))));
}

fn a_payload_cant_be_its_own_parent<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    assert!(matches!(c.insert(vec![new(header(1, s, 10, 10, 0, &[1], &[]))]), Err(CatalogError::Invalid(_))));
    assert!(!present(&c, 1));
}

fn limits_of_zero_are_refused<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    assert!(matches!(c.range(s, iv(0, 1), None, 0), Err(CatalogError::Invalid(_))));
    assert!(matches!(c.descendants(&[id(1)], 0), Err(CatalogError::Invalid(_))));
}

fn a_chunk_must_match_the_frames_it_absorbs<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    c.insert(vec![
        new(header(1, s, 0, 0, 0, &[], &[])),
        new(header(2, s, 33, 33, 0, &[], &[])),
        new(header(3, s, 50, 50, 0, &[2], &[])),
    ])
    .unwrap();
    let chunk = |frames: Vec<ChunkFrame>, loc: Option<StorageLocator>| NewChunk {
        payload: NewPayload { header: header(100, s, 0, 40, 9, &[], &[]), location: loc, coverage: None },
        frames,
    };
    let ok = || vec![ChunkFrame::new(0, 0, 1), ChunkFrame::new(33, 1, 1)];
    let refused = |r: Result<Inserted>| assert!(matches!(r, Err(CatalogError::Invalid(_))), "{r:?}");
    refused(c.commit_chunk(
        chunk(vec![ChunkFrame::new(5, 0, 1), ChunkFrame::new(33, 1, 1)], Some(warm("x"))),
        &[id(1), id(2)],
    ));
    refused(c.commit_chunk(chunk(ok(), Some(warm("x"))), &[id(2), id(1)]));
    refused(c.commit_chunk(chunk(ok(), None), &[id(1), id(2)]));
    refused(c.commit_chunk(chunk(vec![ChunkFrame::new(50, 0, 1)], Some(warm("x"))), &[id(3)]));
    assert!(present(&c, 1) && present(&c, 2) && !present(&c, 100), "nothing was absorbed");
    c.commit_chunk(chunk(ok(), Some(warm("x"))), &[id(1), id(2)]).unwrap();
}

fn keepers_are_remembered_and_a_purges_acknowledgements_are_recorded<C: Catalog>(c: C) {
    assert_eq!(c.keepers().unwrap(), Vec::<String>::new());
    c.add_keeper("vector-worker").unwrap();
    c.add_keeper("tier-worker").unwrap();
    c.add_keeper("tier-worker").unwrap();
    assert_eq!(c.keepers().unwrap(), ["tier-worker", "vector-worker"], "each once, sorted");
    graph(&c);
    c.start_purge(PurgeSelector::Entity(id(1)), "t", "", id(540), 1).unwrap();
    assert!(matches!(c.purge_acked(id(541), "tier-worker"), Err(CatalogError::NotFound(_))));
    assert_eq!(c.purge_record(id(540)).unwrap().unwrap().acked, Vec::<String>::new());
    c.purge_acked(id(540), "tier-worker").unwrap();
    c.purge_acked(id(540), "tier-worker").unwrap();
    assert_eq!(c.purge_record(id(540)).unwrap().unwrap().acked, ["tier-worker"]);
    assert!(c.forget_keeper("tier-worker").unwrap());
    assert!(!c.forget_keeper("tier-worker").unwrap(), "gone already");
    assert_eq!(c.keepers().unwrap(), ["vector-worker"]);
}

fn a_frame_the_worker_cut_is_left_as_it_was_and_the_chunk_lands_without_it<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    c.insert(vec![
        NewPayload { header: header(1, s, 0, 0, 0, &[], &[]), location: Some(shm(1)), coverage: None },
        NewPayload { header: header(2, s, 33, 33, 0, &[], &[]), location: Some(shm(2)), coverage: None },
    ])
    .unwrap();
    // The worker cut frame 2 from the file (a purge reached it before this commit): the frame stays its own payload,
    // with its copies, and the chunk takes frame 1 only.
    let mut frames = vec![ChunkFrame::new(0, 0, 1), ChunkFrame::new(33, 1, 1)];
    frames[1].cut = true;
    let chunk = NewChunk {
        payload: NewPayload { header: header(100, s, 0, 40, 9, &[], &[]), location: Some(warm("x")), coverage: None },
        frames,
    };
    let committed = c.commit_chunk(chunk, &[id(1), id(2)]).unwrap();
    assert_eq!((committed.hidden, committed.cut), (false, vec![1]));
    assert!(matches!(c.get(id(1)).unwrap(), Some(Resolved::AbsorbedFrame { frame_index: 0, .. })), "absorbed");
    match c.get(id(2)).unwrap() {
        Some(Resolved::Payload(p)) => assert_eq!(p.locations, vec![shm(2)], "{p:?}"),
        other => panic!("{other:?}"),
    }
    assert_eq!(ids(&c.range(s, iv(0, 100), None, 10).unwrap()), vec![id(100), id(2)]);
}

fn a_cut_frame_is_marked_and_the_tier_worker_rewrites_the_chunk<C: Catalog>(c: C) {
    let s = chunked(&c);
    c.purge(PurgeSelector::Range { span: iv(35, 40), streams: Some(vec![s]) }, "t", "", id(506), 1).unwrap();
    let frames = |c: &C| match c.get(id(100)).unwrap() {
        Some(Resolved::Payload(p)) => (p.frames.unwrap(), p.locations),
        other => panic!("{other:?}"),
    };
    let (before, _) = frames(&c);
    assert_eq!(before.iter().map(|f| f.cut).collect::<Vec<_>>(), vec![false, true, false], "readers skip frame 1");
    let mut rewritten = before.clone();
    rewritten[1].length = 0;
    rewritten[2].offset = 300;
    c.rewrite_chunk(id(100), rewritten.clone(), warm("file:///c/0b")).unwrap();
    assert_eq!(frames(&c), (rewritten, vec![warm("file:///c/0b")]));
    let mut uncut = before;
    uncut[1].cut = false;
    assert!(matches!(c.rewrite_chunk(id(100), uncut, warm("y")), Err(CatalogError::Invalid(_))));
    assert!(matches!(c.rewrite_chunk(id(1), vec![], warm("y")), Err(CatalogError::NotFound(_))));
}

fn an_entity_purge_record_keeps_no_trace_of_the_entity<C: Catalog>(c: C) {
    graph(&c);
    c.purge(PurgeSelector::Entity(id(900)), "forget", "", id(507), 1).unwrap();
    assert_eq!(c.purge_record(id(507)).unwrap().unwrap().kind, PurgeKind::Entity);
}

fn a_range_purge_of_an_unknown_stream_is_refused<C: Catalog>(c: C) {
    graph(&c);
    let r = c.purge(PurgeSelector::Range { span: iv(0, 100), streams: Some(vec![StreamId(999)]) }, "t", "", id(508), 1);
    assert!(matches!(r, Err(CatalogError::NotFound(_))));
    assert!(present(&c, 1));
}

fn descendants_follow_provenance_not_absorption<C: Catalog>(c: C) {
    chunked(&c);
    assert_eq!(
        c.descendants(&[id(100)], 10).unwrap(),
        Vec::<Uuid>::new(),
        "frames inside a chunk aren't derived from it"
    );
}

fn a_frame_kept_for_its_subjects_still_resolves<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    c.insert(vec![new(header(1, s, 0, 0, 0, &[], &[900])), new(header(2, s, 10, 10, 0, &[], &[]))]).unwrap();
    let chunk = NewChunk {
        payload: NewPayload { header: header(100, s, 0, 20, 5, &[], &[]), location: Some(warm("x")), coverage: None },
        frames: vec![ChunkFrame::new(0, 0, 1), ChunkFrame::new(10, 1, 1)],
    };
    c.commit_chunk(chunk, &[id(1), id(2)]).unwrap();
    assert!(matches!(c.get(id(1)).unwrap(), Some(Resolved::AbsorbedFrame { frame_index: 0, .. })));
    assert!(matches!(c.get(id(2)).unwrap(), Some(Resolved::AbsorbedFrame { frame_index: 1, .. })));
}

fn an_inserts_coverage_mark_merges_on_insert<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    let d = c.register_stream(info("derived")).unwrap();
    c.add_coverage(CoverageMark { source: s, stage: StageId(3), span: iv(0, 10) }).unwrap();
    let mark = CoverageMark { source: s, stage: StageId(3), span: iv(10, 20) };
    c.insert(vec![NewPayload { header: header(1, d, 10, 20, 0, &[], &[]), location: None, coverage: Some(mark) }])
        .unwrap();
    assert_eq!(c.coverage(s, StageId(3), iv(0, 100)).unwrap(), vec![iv(0, 20)]);
}

/// Runs a purge one row at a time, returning each step's delta and the report.
fn step_through<C: Catalog>(c: &C, purge: Uuid) -> (Vec<PurgeDelta>, PurgeReport) {
    let mut deltas = Vec::new();
    loop {
        match c.purge_step(purge, 1).unwrap() {
            PurgeProgress::Running { delta, .. } => deltas.push(delta),
            PurgeProgress::Removed { report, delta } => {
                deltas.push(delta);
                return (deltas, report);
            }
        }
    }
}

fn a_stepped_purge_hides_a_budget_at_a_time_and_says_what_it_hid<C: Catalog>(c: C) {
    let (cam, _) = graph(&c);
    c.add_coverage(CoverageMark { source: cam, stage: StageId(1), span: iv(0, 100) }).unwrap();
    c.start_purge(PurgeSelector::Range { span: iv(10, 11), streams: Some(vec![cam]) }, "t", "", id(510), 5).unwrap();
    assert!(present(&c, 1), "nothing is hidden before the first step");
    assert_eq!(c.pending_purges().unwrap(), vec![id(510)]);
    assert!(matches!(c.complete_purge(id(510), 6), Err(CatalogError::Invalid(_))), "it's still running");
    let first = c.purge_step(id(510), 1).unwrap();
    let PurgeProgress::Running { delta, done: 1, .. } = first else { panic!("{first:?}") };
    assert_eq!(delta.doomed.len(), 1, "one row for a budget of 1");
    let hidden = delta.doomed[0].id;
    assert!(c.get(hidden).unwrap().is_none(), "hidden from the step that found it");
    let (rest, report) = step_through(&c, id(510));
    let mut seen: Vec<Uuid> =
        std::iter::once(hidden).chain(rest.iter().flat_map(|d| d.doomed.iter().map(|x| x.id))).collect();
    seen.sort();
    let mut gone: Vec<Uuid> = report.removed.iter().map(|r| r.id).collect();
    gone.sort();
    assert_eq!(seen, gone, "the deltas name exactly what was removed");
    assert!(rest.len() > 1, "a budget of 1 takes several steps");
    let uncovered: Vec<_> = rest.iter().flat_map(|d| d.uncovered.iter().copied()).collect();
    assert!(uncovered.contains(&(cam, StageId(1), iv(10, 11))), "{uncovered:?}");
    assert_eq!(c.purge_step(id(510), 1).unwrap(), PurgeProgress::Removed { report, delta: PurgeDelta::default() });
}

fn a_payload_derived_from_one_a_purge_hid_is_purged_with_it<C: Catalog>(c: C) {
    let (_, d) = graph(&c);
    c.start_purge(PurgeSelector::Entity(id(900)), "forget", "", id(511), 1).unwrap();
    let mut hidden = Vec::new();
    while hidden.len() < 3 {
        if let PurgeProgress::Running { delta, .. } = c.purge_step(id(511), 1).unwrap() {
            hidden.extend(delta.doomed.into_iter().map(|x| x.id));
        }
    }
    assert!(hidden.contains(&id(21)), "{hidden:?}");
    // A worker finishes a job it started before the purge: derived from the hidden embedding.
    c.insert(vec![new(header(40, d, 10, 10, 9, &[21], &[]))]).unwrap();
    assert!(!present(&c, 40), "hidden as soon as it lands");
    let (deltas, report) = step_through(&c, id(511));
    assert!(deltas.iter().any(|x| x.doomed.iter().any(|x| x.id == id(40))), "and said so in a delta");
    assert!(report.removed.iter().any(|r| r.id == id(40)));
    assert!(present(&c, 1) && !present(&c, 40));
}

fn a_cut_frame_is_in_the_last_steps_delta<C: Catalog>(c: C) {
    let s = chunked(&c);
    c.start_purge(PurgeSelector::Range { span: iv(35, 40), streams: Some(vec![s]) }, "t", "", id(512), 1).unwrap();
    let (deltas, _) = step_through(&c, id(512));
    let cuts: Vec<_> = deltas.iter().flat_map(|d| d.cut_frames.iter().cloned()).collect();
    assert_eq!(cuts, vec![cut_100_1(s)]);
}

fn a_frame_being_cut_stops_being_readable_at_once<C: Catalog>(c: C) {
    let s = chunked(&c);
    c.start_purge(PurgeSelector::Range { span: iv(35, 40), streams: Some(vec![s]) }, "t", "", id(514), 1).unwrap();
    let PurgeProgress::Running { delta, .. } = c.purge_step(id(514), 1).unwrap() else { panic!("one step isn't all") };
    assert_eq!(delta.cut_frames, vec![cut_100_1(s)], "the first step names the cut");
    let frames = match c.get(id(100)).unwrap() {
        Some(Resolved::Payload(p)) => p.frames.unwrap(),
        other => panic!("{other:?}"),
    };
    assert!(frames[1].cut, "readers skip it from the first step on");
}

fn a_payload_being_purged_takes_no_new_copies<C: Catalog>(c: C) {
    let (cam, _) = graph(&c);
    c.start_purge(PurgeSelector::Range { span: iv(10, 11), streams: Some(vec![cam]) }, "t", "", id(515), 1).unwrap();
    let PurgeProgress::Running { delta, .. } = c.purge_step(id(515), 1).unwrap() else { panic!() };
    let hidden = delta.doomed[0].id;
    assert!(matches!(c.set_location(hidden, warm("x")), Err(CatalogError::Conflict(_))));
}

/// Frame 1 of `chunked`'s chunk, as a delta names its cut.
fn cut_100_1(s: StreamId) -> CutFrame {
    CutFrame { chunk: id(100), index: 1, stream: s, slot: iv(33, 66), tiers: vec![StorageTier::Tier1Warm] }
}

/// Every id the deltas name, in order, with repeats.
fn named(deltas: &[PurgeDelta]) -> Vec<Uuid> {
    deltas.iter().flat_map(|d| d.doomed.iter().map(|x| x.id)).collect()
}

fn sorted(mut v: Vec<Uuid>) -> Vec<Uuid> {
    v.sort();
    v
}

fn overlapping_purges_name_each_row_in_exactly_one_delta<C: Catalog>(c: C) {
    // Crop 11 and embedding 21 carry subject 900; give the summary 31's other parent a second subject too.
    let (cam, _) = graph(&c);
    c.start_purge(PurgeSelector::Entity(id(900)), "a", "", id(520), 1).unwrap();
    c.start_purge(PurgeSelector::Range { span: iv(0, 100), streams: Some(vec![cam]) }, "b", "", id(521), 1).unwrap();
    let mut deltas = Vec::new();
    let mut reports = Vec::new();
    let mut open = vec![id(520), id(521)];
    while !open.is_empty() {
        open.retain(|&p| match c.purge_step(p, 1).unwrap() {
            PurgeProgress::Running { delta, .. } => {
                deltas.push(delta);
                true
            }
            PurgeProgress::Removed { report, delta } => {
                deltas.push(delta);
                reports.push(report);
                false
            }
        });
    }
    let names = named(&deltas);
    let unique: std::collections::BTreeSet<Uuid> = names.iter().copied().collect();
    assert_eq!(names.len(), unique.len(), "no row is named twice: {names:?}");
    let removed = sorted(reports.iter().flat_map(|r| r.removed.iter().map(|x| x.id)).collect());
    assert_eq!(sorted(names), removed, "the deltas name exactly what the reports removed");
}

fn a_delta_names_the_copies_a_payload_has_when_it_is_hidden<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    c.insert(vec![NewPayload { header: header(1, s, 10, 10, 0, &[], &[]), location: Some(shm(1)), coverage: None }])
        .unwrap();
    c.start_purge(PurgeSelector::Range { span: iv(0, 100), streams: Some(vec![s]) }, "t", "", id(522), 1).unwrap();
    // The tier worker flushes it and frees its slot before the purge's first step.
    c.set_location(id(1), warm("file:///c/1")).unwrap();
    c.clear_location(id(1), StorageTier::Tier0Hot).unwrap();
    let PurgeProgress::Running { delta, .. } = c.purge_step(id(522), 1).unwrap() else { panic!() };
    assert_eq!(delta.doomed[0].tiers, vec![StorageTier::Tier1Warm]);
    assert!(matches!(c.clear_location(id(1), StorageTier::Tier1Warm), Err(CatalogError::Conflict(_))));
}

fn a_hidden_chunk_is_not_rewritten<C: Catalog>(c: C) {
    let s = chunked(&c);
    c.start_purge(PurgeSelector::Range { span: iv(0, 100), streams: Some(vec![s]) }, "t", "", id(523), 1).unwrap();
    let mut hidden = Vec::new();
    while !hidden.contains(&id(100)) {
        let PurgeProgress::Running { delta, .. } = c.purge_step(id(523), 1).unwrap() else { panic!("ended early") };
        hidden.extend(delta.doomed.into_iter().map(|x| x.id));
    }
    let frames = [0, 33, 66].map(|t| ChunkFrame::new(t, t * 10, 300)).to_vec();
    assert!(matches!(c.rewrite_chunk(id(100), frames, warm("y")), Err(CatalogError::Conflict(_))));
}

fn a_chunk_landing_wholly_inside_a_purge_is_doomed_with_its_frames<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    c.insert((1..=3).map(|n| new(header(n, s, u64::from(n) * 10, u64::from(n) * 10, 0, &[], &[]))).collect()).unwrap();
    c.start_purge(PurgeSelector::Range { span: iv(0, 100), streams: Some(vec![s]) }, "t", "", id(524), 1).unwrap();
    c.purge_step(id(524), 1).unwrap();
    let chunk = NewChunk {
        payload: NewPayload { header: header(100, s, 10, 31, 0, &[], &[]), location: Some(warm("x")), coverage: None },
        frames: [10, 20, 30].map(|t| ChunkFrame::new(t, t, 1)).to_vec(),
    };
    // Every frame is queued for the purge, and the purge's window covers the whole chunk: the commit goes through
    // (nothing refused, so nothing is set aside, and the chunk's key being its hidden first frame's doesn't stop it),
    // the chunk lands hidden, and the purge removes chunk and frames alike, as it would a chunk it found in place.
    let committed = c.commit_chunk(chunk, &[id(1), id(2), id(3)]).unwrap();
    assert!(committed.hidden, "wholly inside the purge's window: doomed on landing");
    let report = loop {
        if let PurgeProgress::Removed { report, .. } = c.purge_step(id(524), 1_000).unwrap() {
            break report;
        }
    };
    assert!(report.removed.iter().any(|r| r.id == id(100)), "{report:?}");
    assert!(!present(&c, 100) && !present(&c, 1) && !present(&c, 2) && !present(&c, 3));
}

fn a_child_of_a_payload_the_purge_has_yet_to_reach_is_purged_too<C: Catalog>(c: C) {
    let (cam, d) = graph(&c);
    c.start_purge(PurgeSelector::Range { span: iv(0, 100), streams: Some(vec![cam]) }, "t", "", id(525), 1).unwrap();
    c.purge_step(id(525), 1).unwrap();
    // Frame 4 isn't hidden yet (budget 1): what's derived from it now is hidden at the latest when the purge reaches
    // it, and purged with it.
    c.insert(vec![new(header(41, d, 40, 40, 0, &[4], &[]))]).unwrap();
    let (deltas, report) = step_through(&c, id(525));
    assert!(named(&deltas).contains(&id(41)) && report.removed.iter().any(|r| r.id == id(41)));
    assert!(!present(&c, 41));
}

fn a_frame_cut_already_is_not_cut_again<C: Catalog>(c: C) {
    let s = chunked(&c);
    c.purge(PurgeSelector::Range { span: iv(35, 40), streams: Some(vec![s]) }, "first", "", id(526), 1).unwrap();
    c.start_purge(PurgeSelector::Range { span: iv(35, 40), streams: Some(vec![s]) }, "again", "", id(527), 1).unwrap();
    let (deltas, report) = step_through(&c, id(527));
    assert!(deltas.iter().all(|d| d.cut_frames.is_empty()), "{deltas:?}");
    assert!(report.partial_chunks.is_empty());
}

fn purge_steps_refuse_what_they_should<C: Catalog>(c: C) {
    graph(&c);
    assert!(matches!(c.purge_step(id(999), 1), Err(CatalogError::NotFound(_))));
    c.start_purge(PurgeSelector::Entity(id(12345)), "nobody", "", id(528), 1).unwrap();
    assert!(matches!(c.purge_step(id(528), 0), Err(CatalogError::Invalid(_))));
    let (_, report) = step_through(&c, id(528));
    assert_eq!((report.removed.len(), report.removed_frames.len()), (0, 0), "a purge that finds nothing ends");
    c.start_purge(PurgeSelector::Entity(id(900)), "x", "", id(529), 2).unwrap();
    assert_eq!(c.pending_purges().unwrap(), vec![id(528), id(529)], "oldest first");
    assert!(matches!(c.coverage_keys(None, 0), Err(CatalogError::Invalid(_))));
}

/// The frames a range purge forgets on a stream of five 10 ns frames over [0, 50), raw or absorbed into one chunk.
fn forgotten_frames<C: Catalog>(c: &C, k: u16, span: Interval, compact: bool) -> Vec<u64> {
    let s = c.register_stream(info(&format!("{}{k}", if compact { "chunked" } else { "raw" }))).unwrap();
    let base = 3_000 + k * 20 + if compact { 10 } else { 0 };
    let frames: Vec<u16> = (0..5).map(|i| base + i).collect();
    c.insert(frames.iter().zip(0u64..).map(|(&n, i)| new(header(n, s, i * 10, i * 10 + 10, 0, &[], &[]))).collect())
        .unwrap();
    if compact {
        let chunk = NewChunk {
            payload: NewPayload {
                header: header(base + 9, s, 0, 50, 1, &[], &[]),
                location: Some(warm("x")),
                coverage: None,
            },
            frames: (0..5).map(|i| ChunkFrame::new(i * 10, i, 1)).collect(),
        };
        c.commit_chunk(chunk, &frames.iter().map(|&n| id(n)).collect::<Vec<_>>()).unwrap();
    }
    let report = c.purge(PurgeSelector::Range { span, streams: Some(vec![s]) }, "t", "", id(base + 8), 1).unwrap();
    let mut starts: Vec<u64> = if compact {
        let whole = report.removed.iter().any(|r| r.id == id(base + 9));
        let cut = report.partial_chunks.iter().flat_map(|p| p.frames.iter().map(|&i| u64::from(i) * 10));
        if whole { (0..5).map(|i| i * 10).collect() } else { cut.collect() }
    } else {
        report.removed.iter().map(|r| frames.iter().position(|&n| id(n) == r.id).unwrap() as u64 * 10).collect()
    };
    starts.sort_unstable();
    starts
}

fn a_range_purge_forgets_the_same_frames_before_and_after_compaction<C: Catalog>(c: C) {
    // Every range on a 5 ns grid over [0, 60), plus a few off it. The repro is among them: [15, 25) overlaps the
    // frames at 10 and 20, flushed or not.
    let mut spans: Vec<(u64, u64)> = (0..12u64).flat_map(|a| (a + 1..13).map(move |b| (a * 5, b * 5))).collect();
    spans.extend([(9, 11), (19, 20), (20, 21), (49, 50)]);
    for (k, &(lo, hi)) in (0u16..).zip(&spans) {
        let span = iv(lo, hi);
        let raw = forgotten_frames(&c, k, span, false);
        assert_eq!(raw, forgotten_frames(&c, k, span, true), "purging {span:?}");
        if (lo, hi) == (15, 25) {
            assert_eq!(raw, vec![10, 20]);
        }
    }
}

/// Five 10 ns frames on a fresh stream over [0, 50), and a range purge of [0, 100) over it, stepped `k` times with a
/// budget of 1 before the tier worker commits a chunk of the last three frames and a camera adds a frame at 5. The
/// chunk commits whatever the purge has done so far (its frames hidden, removed, or reported), and after the purge
/// nothing on the stream is readable.
fn commits_after_k_steps_are_purged_too<C: Catalog>(c: &C, k: usize) {
    let s = c.register_stream(info(&format!("cam-k{k}"))).unwrap();
    let base = 5_000 + u16::try_from(k).unwrap() * 20;
    let frames: Vec<u16> = (0..5).map(|i| base + i).collect();
    c.insert(frames.iter().zip(0u64..).map(|(&n, i)| new(header(n, s, i * 10, i * 10 + 10, 0, &[], &[]))).collect())
        .unwrap();
    let purge = id(base + 19);
    c.start_purge(PurgeSelector::Range { span: iv(0, 100), streams: Some(vec![s]) }, "t", "", purge, 1).unwrap();
    for _ in 0..k {
        c.purge_step(purge, 1).unwrap();
    }
    // Once the purge has finished, what lands later is new data, not something the purge must take.
    let running = c.purge_record(purge).unwrap().unwrap().report.is_none();
    let chunk = NewChunk {
        payload: NewPayload {
            header: header(base + 10, s, 20, 50, 1, &[], &[]),
            location: Some(warm("x")),
            coverage: None,
        },
        frames: (2..5).map(|i| ChunkFrame::new(i * 10, i, 1)).collect(),
    };
    let committed = c
        .commit_chunk(chunk, &frames[2..].iter().map(|&n| id(n)).collect::<Vec<_>>())
        .unwrap_or_else(|e| panic!("after {k} steps the chunk was refused: {e}"));
    assert!(
        running || committed.cut == [0, 1, 2],
        "after {k} steps, a finished purge's frames weren't cut: {committed:?}"
    );
    let late_frame = c.insert(vec![new(header(base + 11, s, 5, 6, 2, &[], &[]))]);
    assert!(late_frame.is_ok() || matches!(late_frame, Err(CatalogError::Conflict(_))), "{late_frame:?}");
    step_through(c, purge);
    if !running {
        assert!(frames.iter().all(|&n| !present(c, n)), "after {k} steps, a frame survived the finished purge");
        return;
    }
    let page = c.range(s, iv(0, 100), None, 100).unwrap();
    for item in &page.items {
        let frames = item.frames.as_ref().unwrap_or_else(|| panic!("after {k} steps, {:?} survived", item.header));
        assert!(frames.iter().all(|f| f.cut), "after {k} steps, a chunk kept uncut frames: {frames:?}");
    }
    for n in frames.iter().chain([&(base + 11)]) {
        assert!(!present(c, *n), "after {k} steps, frame {n} survived");
    }
}

fn a_range_purge_takes_what_is_committed_at_any_point_of_its_scan<C: Catalog>(c: C) {
    // Past the redb catalog's last step for five frames (it removes and cuts in phases after hiding), so a commit
    // lands during each phase and after the purge has finished.
    for k in 0..=20 {
        commits_after_k_steps_are_purged_too(&c, k);
    }
}

fn a_chunk_over_the_frame_limit_is_refused<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    let n = u16::try_from(MAX_CHUNK_FRAMES + 1).unwrap();
    c.insert((1..=n).map(|i| new(header(i, s, u64::from(i), u64::from(i), 0, &[], &[]))).collect()).unwrap();
    let chunk = NewChunk {
        payload: NewPayload {
            header: header(60_000, s, 1, u64::from(n), 1, &[], &[]),
            location: Some(warm("x")),
            coverage: None,
        },
        frames: (1..=n).map(|i| ChunkFrame::new(u64::from(i), 0, 1)).collect(),
    };
    let absorbed: Vec<Uuid> = (1..=n).map(id).collect();
    assert!(matches!(c.commit_chunk(chunk, &absorbed), Err(CatalogError::Invalid(_))));
}

fn commit_sequences_rise_and_are_never_reused<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    assert_eq!(c.head_seq().unwrap(), 0);
    let first: Vec<u64> = c
        .insert(vec![new(header(1, s, 10, 10, 0, &[], &[900])), new(header(2, s, 20, 20, 0, &[], &[]))])
        .unwrap()
        .iter()
        .map(|i| i.seq)
        .collect();
    assert!(first[0] < first[1] && first[1] == c.head_seq().unwrap(), "{first:?}");
    let Some(Resolved::Payload(p)) = c.get(id(1)).unwrap() else { panic!() };
    assert_eq!(p.commit_seq, first[0]);
    c.purge(PurgeSelector::Entity(id(900)), "t", "", id(600), 1).unwrap();
    let after = c.insert(vec![new(header(3, s, 30, 30, 0, &[], &[]))]).unwrap();
    assert!(after[0].seq > first[1], "a purge never makes a sequence come round again");
}

fn inserting_the_same_payload_again_returns_its_sequence<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    let seq = c.insert(vec![new(header(1, s, 10, 10, 0, &[], &[]))]).unwrap()[0].seq;
    let head = c.head_seq().unwrap();
    // A batch with the duplicate, a new payload, and the new one twice.
    let again = c
        .insert(vec![
            new(header(1, s, 10, 10, 0, &[], &[])),
            new(header(2, s, 20, 20, 0, &[], &[])),
            new(header(2, s, 20, 20, 0, &[], &[])),
        ])
        .unwrap();
    let got: Vec<(u64, bool)> = again.iter().map(|i| (i.seq, i.duplicate)).collect();
    assert_eq!(got, vec![(seq, true), (head + 1, false), (head + 1, true)]);
    assert_eq!(c.head_seq().unwrap(), head + 1, "the duplicate wrote nothing");
    assert!(matches!(c.insert(vec![new(header(1, s, 11, 11, 0, &[], &[]))]), Err(CatalogError::Conflict(_))));
}

fn a_query_walks_a_stream_in_commit_order_with_filters_and_a_scan_cap<C: Catalog>(c: C) {
    let s = c.register_stream(info("events:cam0")).unwrap();
    let other = c.register_stream(info("events:cam1")).unwrap();
    let mut batch: Vec<NewPayload> = (1..=10u16)
        .map(|n| new(header(n, s, u64::from(n) * 10, u64::from(n) * 10, 0, &[], if n % 2 == 0 { &[900] } else { &[] })))
        .collect();
    batch.push(new(header(50, other, 5, 5, 0, &[], &[900])));
    let seqs: Vec<u64> = c.insert(batch).unwrap().iter().map(|i| i.seq).collect();
    let all = QueryFilter::default();
    let ids = |p: &SeqPage| p.items.iter().map(|i| i.header.id()).collect::<Vec<_>>();
    let page = c.query(s, 0, &all, 4, 100).unwrap();
    assert_eq!(ids(&page), (1..=4).map(id).collect::<Vec<_>>());
    assert_eq!(page.next_after, Some(seqs[3]));
    let rest = c.query(s, seqs[3], &all, 100, 100).unwrap();
    assert_eq!((ids(&rest).len(), rest.next_after), (6, None));
    let theirs = QueryFilter { subjects: Some(vec![id(900)]), start_before: Some(75), ..QueryFilter::default() };
    assert_eq!(ids(&c.query(s, 0, &theirs, 100, 100).unwrap()), vec![id(2), id(4), id(6)]);
    let listed = QueryFilter { ids: Some(vec![id(3), id(9), id(50)]), ..QueryFilter::default() };
    assert_eq!(ids(&c.query(s, 0, &listed, 100, 100).unwrap()), vec![id(3), id(9)], "one stream only");
    // A selective filter stays bounded: three rows looked at, nothing found yet, and where to go on.
    let capped = c.query(s, 0, &QueryFilter { ids: Some(vec![id(9)]), ..QueryFilter::default() }, 10, 3).unwrap();
    assert_eq!((capped.items.len(), capped.next_after), (0, Some(seqs[2])));
    assert!(matches!(c.query(s, 0, &all, 0, 1), Err(CatalogError::Invalid(_))));
    assert!(matches!(c.query(StreamId(999), 0, &all, 1, 1), Err(CatalogError::NotFound(_))));
}

fn a_purge_by_ids_takes_them_and_what_came_from_them<C: Catalog>(c: C) {
    graph(&c);
    let report =
        c.purge(PurgeSelector::Ids(vec![id(12), id(99)]), "events", r#"{"stream":"camera:cam0"}"#, id(601), 1).unwrap();
    let mut gone: Vec<Uuid> = report.removed.iter().map(|r| r.id).collect();
    gone.sort();
    assert_eq!(gone, vec![id(12), id(22), id(31)], "the crop, its embedding, and the summary built on it");
    let record = c.purge_record(id(601)).unwrap().unwrap();
    assert_eq!((record.kind, record.detail.as_str()), (PurgeKind::Ids { count: 2 }, r#"{"stream":"camera:cam0"}"#));
}

fn purge_records_list_in_start_order<C: Catalog>(c: C) {
    graph(&c);
    for (n, why) in [(610u16, "a"), (611, "b"), (612, "c")] {
        c.purge(PurgeSelector::Ids(vec![id(99)]), why, "", id(n), u64::from(n)).unwrap();
    }
    let reasons = |l: Vec<PurgeSummary>| l.into_iter().map(|r| r.reason).collect::<Vec<_>>();
    assert_eq!(reasons(c.purges(None, 2).unwrap()), vec!["a", "b"]);
    assert_eq!(reasons(c.purges(Some(id(611)), 10).unwrap()), vec!["c"]);
    assert!(matches!(c.purges(Some(id(999)), 1), Err(CatalogError::NotFound(_))));
    assert!(matches!(c.purges(None, 0), Err(CatalogError::Invalid(_))));
}

fn small_payloads_keep_their_bytes_inline<C: Catalog>(c: C) {
    let s = c.register_stream(info("events:cam0")).unwrap();
    let body = StorageLocator::inline(br#"{"type":"face.seen"}"#.to_vec()).unwrap();
    c.insert(vec![NewPayload {
        header: header(1, s, 10, 10, 0, &[], &[]),
        location: Some(body.clone()),
        coverage: None,
    }])
    .unwrap();
    let Some(Resolved::Payload(p)) = c.get(id(1)).unwrap() else { panic!() };
    assert_eq!(p.locations, vec![body]);
    let too_big = StorageLocator::Inline { bytes: vec![0; continuum_core::MAX_INLINE_BYTES + 1] };
    let refused = c.insert(vec![NewPayload {
        header: header(2, s, 20, 20, 0, &[], &[]),
        location: Some(too_big),
        coverage: None,
    }]);
    assert!(matches!(refused, Err(CatalogError::Invalid(_))));
}

fn a_chunk_takes_a_new_sequence_and_its_frames_leave_the_query<C: Catalog>(c: C) {
    let s = chunked(&c);
    let page = c.query(s, 0, &QueryFilter::default(), 100, 100).unwrap();
    assert_eq!(page.items.iter().map(|i| i.header.id()).collect::<Vec<_>>(), vec![id(100)]);
    assert_eq!(page.items[0].commit_seq, c.head_seq().unwrap(), "the chunk was the last commit");
}

fn a_payload_that_lands_under_a_purge_says_so_and_its_delta_entry_is_late<C: Catalog>(c: C) {
    let (_, d) = graph(&c);
    c.start_purge(PurgeSelector::Entity(id(900)), "forget", "", id(620), 1).unwrap();
    let mut hidden = Vec::new();
    while !hidden.contains(&id(21)) {
        let PurgeProgress::Running { delta, .. } = c.purge_step(id(620), 1).unwrap() else { panic!("ended early") };
        assert!(delta.doomed.iter().all(|x| !x.late), "found by the purge itself, not late");
        hidden.extend(delta.doomed.into_iter().map(|x| x.id));
    }
    let landed = c.insert(vec![new(header(40, d, 10, 10, 9, &[21], &[]))]).unwrap();
    assert!(landed[0].hidden && !landed[0].duplicate, "{landed:?}");
    let again = c.insert(vec![new(header(40, d, 10, 10, 9, &[21], &[]))]).unwrap();
    assert!(again[0].hidden && again[0].duplicate && again[0].seq == landed[0].seq);
    let (deltas, _) = step_through(&c, id(620));
    let entry = deltas.iter().flat_map(|x| x.doomed.iter()).find(|x| x.id == id(40)).unwrap();
    assert!(entry.late);
    let fresh = c.insert(vec![new(header(41, d, 10, 10, 10, &[1], &[]))]).unwrap();
    assert!(!fresh[0].hidden, "nothing is purging what it came from");
}

fn a_payload_landing_where_a_purge_reaches_is_hidden_on_landing<C: Catalog>(c: C) {
    let (cam, d) = graph(&c);
    // A range purge over cam0, stepped once: a frame landing in its range is hidden as it lands, on either catalog.
    c.start_purge(PurgeSelector::Range { span: iv(0, 100), streams: Some(vec![cam]) }, "t", "", id(630), 1).unwrap();
    c.purge_step(id(630), 1).unwrap();
    let in_range = c.insert(vec![new(header(60, cam, 55, 55, 0, &[], &[]))]).unwrap();
    assert!(in_range[0].hidden, "{in_range:?}");
    let (deltas, _) = step_through(&c, id(630));
    assert!(deltas.iter().flat_map(|x| x.doomed.iter()).any(|x| x.id == id(60) && x.late));
    // An entity purge: a payload about the subject is hidden as it lands, whoever its parents are.
    c.start_purge(PurgeSelector::Entity(id(901)), "t", "", id(631), 1).unwrap();
    let about = c.insert(vec![new(header(61, d, 70, 70, 0, &[], &[901]))]).unwrap();
    assert!(about[0].hidden);
    let (deltas, report) = step_through(&c, id(631));
    assert!(deltas.iter().flat_map(|x| x.doomed.iter()).any(|x| x.id == id(61) && x.late));
    assert!(report.removed.iter().any(|r| r.id == id(61)));
}

fn inline_bytes_never_move_and_a_purge_report_keeps_none<C: Catalog>(c: C) {
    let s = c.register_stream(info("events:cam0")).unwrap();
    let secret = br#"{"said":"the alarm code is 4711"}"#.to_vec();
    let body = StorageLocator::inline(secret.clone()).unwrap();
    c.insert(vec![NewPayload { header: header(1, s, 10, 10, 0, &[], &[]), location: Some(body), coverage: None }])
        .unwrap();
    assert!(matches!(c.set_location(id(1), warm("x")), Err(CatalogError::Invalid(_))), "a chunk copy would replace it");
    assert!(matches!(c.clear_location(id(1), StorageTier::Tier1Warm), Err(CatalogError::Invalid(_))));
    c.insert(vec![new(header(2, s, 20, 20, 0, &[], &[]))]).unwrap();
    let inline_again = StorageLocator::inline(b"x".to_vec()).unwrap();
    assert!(
        matches!(c.set_location(id(2), inline_again), Err(CatalogError::Invalid(_))),
        "inline is set at insert only"
    );
    let report = c.purge(PurgeSelector::Ids(vec![id(1)]), "forget", "", id(632), 1).unwrap();
    let kept = &report.removed[0].locations;
    assert_eq!(kept, &vec![StorageLocator::Inline { bytes: Vec::new() }], "it was inline, and the bytes are gone");
    let record = c.purge_record(id(632)).unwrap().unwrap().report.unwrap();
    assert_eq!(record.removed[0].locations, *kept);
    let listed = c.purges(None, 10).unwrap();
    assert_eq!((listed[0].removed, listed[0].finished), (1, true));
}

fn a_chunk_cant_be_placed_inline<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    c.insert(vec![new(header(1, s, 0, 0, 0, &[], &[]))]).unwrap();
    let chunk = NewChunk {
        payload: NewPayload {
            header: header(100, s, 0, 9, 1, &[], &[]),
            location: StorageLocator::inline(b"frames".to_vec()),
            coverage: None,
        },
        frames: vec![ChunkFrame::new(0, 0, 1)],
    };
    assert!(matches!(c.commit_chunk(chunk, &[id(1)]), Err(CatalogError::Invalid(_))));
}

fn a_scrub_runs_only_when_due<C: Catalog>(c: C) {
    assert!(!c.scrub_needed().unwrap());
    assert_eq!(c.scrub_step(16).unwrap(), ScrubProgress::Idle);
    graph(&c);
    c.purge(PurgeSelector::Entity(id(900)), "forget", "", id(513), 1).unwrap();
    assert!(c.scrub_needed().unwrap(), "a finished purge makes one due");
    let mut last = ScrubProgress::Idle;
    for _ in 0..1000 {
        last = c.scrub_step(16).unwrap();
        if last == ScrubProgress::Swapped {
            break;
        }
    }
    assert_eq!(last, ScrubProgress::Swapped);
    assert!(!c.scrub_needed().unwrap());
    assert_eq!(c.scrub_step(16).unwrap(), ScrubProgress::Idle);
    c.request_scrub().unwrap();
    assert!(c.scrub_needed().unwrap());
    assert!(matches!(c.scrub_step(0), Err(CatalogError::Invalid(_))));
    assert!(present(&c, 1), "a scrub keeps every live row");
}

fn streams_and_coverage_keys_page_in_order<C: Catalog>(c: C) {
    let ids: Vec<StreamId> = ["a", "b", "c"].iter().map(|n| c.register_stream(info(n)).unwrap()).collect();
    let first = c.streams(None, 2).unwrap();
    assert_eq!(first.iter().map(|(s, _)| *s).collect::<Vec<_>>(), ids[..2]);
    assert_eq!(first[0].1.name, "a");
    let rest = c.streams(Some(first[1].0), 2).unwrap();
    assert_eq!(rest.iter().map(|(s, _)| *s).collect::<Vec<_>>(), ids[2..]);
    for (s, g) in [(ids[1], 4), (ids[0], 9), (ids[0], 2)] {
        c.add_coverage(CoverageMark { source: s, stage: StageId(g), span: iv(0, 1) }).unwrap();
    }
    let keys = c.coverage_keys(None, 2).unwrap();
    assert_eq!(keys, vec![(ids[0], StageId(2)), (ids[0], StageId(9))]);
    assert_eq!(c.coverage_keys(Some(keys[1]), 10).unwrap(), vec![(ids[1], StageId(4))]);
    assert!(matches!(c.streams(None, 0), Err(CatalogError::Invalid(_))));
}

fn readers_run_alongside_a_writer<C: Catalog + 'static>(c: C) {
    let c = Arc::new(c);
    let s = c.register_stream(info("cam0")).unwrap();
    let readers: Vec<_> = (0..4)
        .map(|_| {
            let c = Arc::clone(&c);
            std::thread::spawn(move || {
                for _ in 0..200 {
                    let page = c.range(s, iv(0, 10_000), None, 1_000).unwrap();
                    let keys: Vec<(u64, u32)> =
                        page.items.iter().map(|i| (i.header.start_ns(), i.header.seq())).collect();
                    assert!(keys.windows(2).all(|w| w[0] < w[1]), "a read sees a consistent, ordered snapshot");
                }
            })
        })
        .collect();
    for n in 0..200u16 {
        c.insert(vec![new(header(n + 1, s, u64::from(n) * 10, u64::from(n) * 10, 0, &[], &[]))]).unwrap();
    }
    for r in readers {
        r.join().unwrap();
    }
    assert_eq!(c.range(s, iv(0, 10_000), None, 1_000).unwrap().items.len(), 200);
}

fn small_tables_keep_what_they_are_given<C: Catalog>(c: C) {
    c.set_cursor("journal", b"pos-1").unwrap();
    c.set_cursor("journal", b"pos-2").unwrap();
    assert_eq!(c.cursor("journal").unwrap(), Some(b"pos-2".to_vec()));
    assert_eq!(c.cursor("other").unwrap(), None);
    assert_eq!(c.load_cost_models().unwrap(), None);
    c.save_cost_models(b"model").unwrap();
    assert_eq!(c.load_cost_models().unwrap(), Some(b"model".to_vec()));
    assert!(!c.is_forgotten(ForgottenHash([1; 32])).unwrap());
    c.forget(ForgottenHash([1; 32])).unwrap();
    assert!(c.is_forgotten(ForgottenHash([1; 32])).unwrap());
}

fn clock_near_finds_the_points_on_each_side_on_that_boot<C: Catalog>(c: C) {
    let (a, b) = (BootId([1; 16]), BootId([2; 16]));
    for (mono, utc) in [(100, 1_000), (200, 2_000), (300, 3_000)] {
        c.record_clock(a, ClockPoint { mono_ns: mono, utc_ns: utc }).unwrap();
    }
    c.record_clock(b, ClockPoint { mono_ns: 250, utc_ns: 9 }).unwrap();
    let near = c.clock_near(a, 250).unwrap();
    assert_eq!(near.at_or_before, Some(ClockPoint { mono_ns: 200, utc_ns: 2_000 }));
    assert_eq!(near.after, Some(ClockPoint { mono_ns: 300, utc_ns: 3_000 }));
    assert_eq!(c.clock_near(a, 200).unwrap().at_or_before.unwrap().mono_ns, 200, "at counts as before");
    assert_eq!(
        c.clock_near(a, 50).unwrap(),
        ClockNear { at_or_before: None, after: Some(ClockPoint { mono_ns: 100, utc_ns: 1_000 }) }
    );
    assert_eq!(c.clock_near(BootId([3; 16]), 250).unwrap(), ClockNear::default(), "another boot's points don't count");
}

fn old_frames_nothing_names_lose_their_rows_and_the_rest_keep_them<C: Catalog>(c: C) {
    // Chunk 100 (0 to 99) absorbed frames 1, 2, and 3; the crop 10 came from frame 2.
    let s = chunked(&c);
    let swept = c.prune_frames(100, None, 10).unwrap();
    assert_eq!(swept, PruneProgress { scanned: 3, pruned: 2, next: None });
    assert!(!present(&c, 1) && !present(&c, 3), "frames nothing came from lose their rows");
    assert!(present(&c, 2), "a frame with a child keeps its row");
    assert_eq!(c.parents(id(10)).unwrap(), vec![id(2)], "lineage is unchanged");
    assert_eq!(ids(&c.range(s, iv(0, 200), None, 10).unwrap()), vec![id(100)], "the chunk still holds them by time");
    // Past the window, a child naming a pruned frame is refused as an unknown parent.
    let d = c.register_stream(info("late")).unwrap();
    assert!(matches!(c.insert(vec![new(header(11, d, 0, 0, 0, &[1], &[]))]), Err(CatalogError::Invalid(_))));
    assert_eq!(c.prune_frames(100, None, 10).unwrap().pruned, 0, "a second sweep finds nothing more");
}

fn frames_inside_the_window_or_with_subjects_keep_their_rows<C: Catalog>(c: C) {
    let s = c.register_stream(info("cam0")).unwrap();
    c.insert(vec![new(header(1, s, 0, 0, 0, &[], &[900])), new(header(2, s, 10, 10, 0, &[], &[]))]).unwrap();
    let chunk = NewChunk {
        payload: NewPayload { header: header(100, s, 0, 20, 5, &[], &[]), location: Some(warm("x")), coverage: None },
        frames: vec![ChunkFrame::new(0, 0, 1), ChunkFrame::new(10, 1, 1)],
    };
    c.commit_chunk(chunk, &[id(1), id(2)]).unwrap();
    assert_eq!(c.prune_frames(20, None, 10).unwrap().pruned, 0, "the chunk ends at 20: not past a cutoff of 20");
    assert_eq!(c.prune_frames(21, None, 10).unwrap().pruned, 1);
    assert!(present(&c, 1), "a frame with a subject keeps its row, so a purge by that entity still finds it");
    assert!(!present(&c, 2));
}

fn a_prune_sweep_goes_a_budget_at_a_time_from_its_cursor<C: Catalog>(c: C) {
    chunked(&c);
    let (mut after, mut pruned, mut steps) = (None, 0, 0);
    loop {
        let step = c.prune_frames(100, after, 1).unwrap();
        assert!(step.scanned <= 1);
        pruned += step.pruned;
        steps += 1;
        match step.next {
            Some(next) => after = Some(next),
            None => break,
        }
    }
    assert_eq!((pruned, steps), (2, 4), "three frames a step each, then an empty step that ends the sweep");
    assert!(matches!(c.prune_frames(100, None, 0), Err(CatalogError::Invalid(_))));
}

fn a_frame_a_running_purge_holds_is_left_to_the_purge<C: Catalog>(c: C) {
    let s = chunked(&c);
    c.start_purge(PurgeSelector::Range { span: iv(0, 1), streams: Some(vec![s]) }, "t", "", id(505), 1).unwrap();
    let PurgeProgress::Running { .. } = c.purge_step(id(505), 1).unwrap() else { panic!("a budget of 1 runs on") };
    assert_eq!(c.prune_frames(100, None, 10).unwrap().pruned, 1, "frame 3 only: frame 1 is the purge's, 2 has a child");
    let (_, report) = step_through(&c, id(505));
    assert_eq!(report.removed_frames, vec![id(1)], "the purge still removes the frame it holds");
}

fn worker(inputs: &[(&str, u64)], position: u64, body: &[u8]) -> WorkerState {
    WorkerState {
        owner: "see-activity".into(),
        contract: "activity-v6".into(),
        position,
        created_ns: 1_000 + position,
        inputs: inputs.iter().map(|(s, r)| ((*s).to_owned(), *r)).collect(),
        body: body.to_vec(),
    }
}

fn purge_one<C: Catalog>(c: &C, payload: u16, purge: u16) {
    c.start_purge(PurgeSelector::Ids(vec![id(payload)]), "test", "", id(purge), 1).unwrap();
    while !matches!(c.purge_step(id(purge), 64).unwrap(), PurgeProgress::Removed { .. }) {}
}

fn a_worker_state_is_kept_in_place_and_survives_appends<C: Catalog>(c: C) {
    let (cam, _) = graph(&c);
    c.register_stream(info("other")).unwrap();
    assert_eq!(c.load_worker_state("see-activity", "activity-v6").unwrap(), None);
    c.save_worker_state(&worker(&[("cam0", 0), ("other", 0)], 7, b"first")).unwrap();
    let second = worker(&[("cam0", 0), ("other", 0)], 9, b"second");
    c.save_worker_state(&second).unwrap();
    c.insert(vec![new(header(90, cam, 90, 90, 0, &[], &[]))]).unwrap();
    assert_eq!(c.load_worker_state("see-activity", "activity-v6").unwrap(), Some(second), "an append isn't a change");
    assert_eq!(c.load_worker_state("see-activity", "activity-v5").unwrap(), None, "another contract is another row");
}

fn a_worker_state_read_before_a_purge_is_refused<C: Catalog>(c: C) {
    graph(&c);
    purge_one(&c, 1, 600);
    let revision = c.invalidation_revision(c.stream_id("cam0").unwrap().unwrap()).unwrap().unwrap();
    assert!(revision > 0);
    let stale = c.save_worker_state(&worker(&[("cam0", revision - 1)], 7, b"x"));
    assert!(matches!(stale, Err(CatalogError::Conflict(_))), "{stale:?}");
    let unknown = c.save_worker_state(&worker(&[("nowhere", 0)], 7, b"x"));
    assert!(matches!(unknown, Err(CatalogError::Conflict(_))), "{unknown:?}");
    assert_eq!(c.load_worker_state("see-activity", "activity-v6").unwrap(), None);
    c.save_worker_state(&worker(&[("cam0", revision)], 7, b"x")).unwrap();
    assert!(c.load_worker_state("see-activity", "activity-v6").unwrap().is_some());
}

fn a_purge_of_an_input_deletes_the_state_built_from_it<C: Catalog>(c: C) {
    graph(&c);
    c.register_stream(info("other")).unwrap();
    c.save_worker_state(&worker(&[("cam0", 0)], 7, b"from cam0")).unwrap();
    let mut elsewhere = worker(&[("other", 0)], 3, b"from other");
    elsewhere.owner = "see-zones".into();
    c.save_worker_state(&elsewhere).unwrap();
    purge_one(&c, 1, 600);
    assert_eq!(c.load_worker_state("see-activity", "activity-v6").unwrap(), None);
    assert!(!c.forget_worker_state("see-activity", "activity-v6").unwrap(), "the purge deleted the row itself");
    assert_eq!(c.load_worker_state("see-zones", "activity-v6").unwrap(), Some(elsewhere), "an unrelated input");
}

fn a_resaved_state_leaves_the_inputs_it_no_longer_names<C: Catalog>(c: C) {
    graph(&c);
    c.register_stream(info("other")).unwrap();
    c.save_worker_state(&worker(&[("cam0", 0)], 7, b"from cam0")).unwrap();
    let now = worker(&[("other", 0)], 8, b"from other");
    c.save_worker_state(&now).unwrap();
    purge_one(&c, 1, 600);
    assert_eq!(c.load_worker_state("see-activity", "activity-v6").unwrap(), Some(now));
}

fn worker_states_are_bounded_and_forgotten<C: Catalog>(c: C) {
    graph(&c);
    let big = worker(&[("cam0", 0)], 1, &vec![0; MAX_WORKER_STATE_BYTES + 1]);
    assert!(matches!(c.save_worker_state(&big), Err(CatalogError::Invalid(_))));
    assert!(matches!(c.save_worker_state(&worker(&[], 1, b"x")), Err(CatalogError::Invalid(_))));
    let unsorted = worker(&[("derived", 0), ("cam0", 0)], 1, b"x");
    assert!(matches!(c.save_worker_state(&unsorted), Err(CatalogError::Invalid(_))));
    assert!(matches!(c.load_worker_state("", "c"), Err(CatalogError::Invalid(_))));
    c.save_worker_state(&worker(&[("cam0", 0)], 1, &vec![7; MAX_WORKER_STATE_BYTES])).unwrap();
    assert!(c.forget_worker_state("see-activity", "activity-v6").unwrap());
    assert_eq!(c.load_worker_state("see-activity", "activity-v6").unwrap(), None);
    assert!(!c.forget_worker_state("see-activity", "activity-v6").unwrap());
}

fn an_age_out_keeps_the_worker_state_and_dependencies_read_before_it<C: Catalog>(c: C) {
    // Aging isn't forgetting: what was built from an aged row is still right, so a worker's saved state and
    // a dependency bound before an age-out stay good. Readers still see the stream change.
    let (cam, _) = graph(&c);
    let progress = c.register_stream(info("progress")).unwrap();
    let read = c.invalidation_revision(cam).unwrap().unwrap();
    c.save_worker_state(&worker(&[("cam0", read)], 7, b"before")).unwrap();
    c.mark_commit_clock(100).unwrap();
    let by = c.commit_seq_at(100).unwrap();
    let aged = PurgeSelector::AgeOut { ids: vec![id(1)], committed_by_seq: by, merge_gap_ns: 0 };
    c.purge(aged, "aged", "", id(840), 50).unwrap();
    let after_aging = c.invalidation_revision(cam).unwrap().unwrap();
    assert!(after_aging > read, "readers' caches still see the stream change");
    assert_eq!(c.forget_revision(cam).unwrap(), Some(read), "an age-out isn't a forget");
    assert!(c.load_worker_state("see-activity", "activity-v6").unwrap().is_some(), "the state outlives an age-out");
    let early = Dependency { stream: cam, span: iv(0, 25), revision: read };
    c.insert_dependent(vec![new(header(640, progress, 5, 5, 0, &[], &[]))], vec![(id(640), vec![early])]).unwrap();
    c.save_worker_state(&worker(&[("cam0", read)], 8, b"saved late")).unwrap();
    // A forget still refuses and drops everything read before it, aged revisions included.
    purge_one(&c, 2, 841);
    assert_eq!(c.load_worker_state("see-activity", "activity-v6").unwrap(), None, "a purge drops the state");
    for revision in [read, after_aging] {
        let stale = Dependency { stream: cam, span: iv(0, 25), revision };
        let r = c.insert_dependent(vec![new(header(641, progress, 5, 5, 1, &[], &[]))], vec![(id(641), vec![stale])]);
        assert!(matches!(r, Err(CatalogError::Conflict(_))), "{revision}: {r:?}");
        let r = c.save_worker_state(&worker(&[("cam0", revision)], 9, b"stale"));
        assert!(matches!(r, Err(CatalogError::Conflict(_))), "{revision}: {r:?}");
    }
    let now = c.invalidation_revision(cam).unwrap().unwrap();
    assert_eq!(c.forget_revision(cam).unwrap(), Some(now), "a purge is");
    c.save_worker_state(&worker(&[("cam0", now)], 9, b"fresh")).unwrap();
    let ahead = Dependency { stream: cam, span: iv(0, 25), revision: now + 1 };
    let r = c.insert_dependent(vec![new(header(642, progress, 5, 5, 2, &[], &[]))], vec![(id(642), vec![ahead])]);
    assert!(matches!(r, Err(CatalogError::Conflict(_))), "a revision not yet reached: {r:?}");
}

fn forgotten_hashes_page_lexicographically_without_repeats(c: impl Catalog) {
    assert_eq!(c.list_forgotten(None, 2).unwrap(), ForgottenPage { hashes: vec![], next: None });
    for hash in [ForgottenHash([255; 32]), ForgottenHash([0; 32]), ForgottenHash([2; 32]), ForgottenHash([1; 32])] {
        c.forget(hash).unwrap();
    }
    let first = c.list_forgotten(None, 2).unwrap();
    assert_eq!(first.hashes, vec![ForgottenHash([0; 32]), ForgottenHash([1; 32])]);
    assert_eq!(first.next, Some(ForgottenHash([1; 32])));
    let last = c.list_forgotten(first.next, 2).unwrap();
    assert_eq!(last.hashes, vec![ForgottenHash([2; 32]), ForgottenHash([255; 32])]);
    assert_eq!(last.next, None);
    assert!(c.list_forgotten(Some(ForgottenHash([255; 32])), 2).unwrap().hashes.is_empty());
    assert_eq!(c.list_forgotten(Some(ForgottenHash([1; 32])), 1).unwrap().next, Some(ForgottenHash([2; 32])));
    assert!(c.list_forgotten(None, 0).is_err());
    assert!(c.list_forgotten(None, 1025).is_err());
    assert!(c.is_forgotten(ForgottenHash([0; 32])).unwrap());
}

macro_rules! conformance {
    ($module:ident, $make:expr) => {
        mod $module {
            use super::*;
            conformance!(@tests $make;
                invalidation_revisions_follow_cross_stream_deletions_and_frame_cuts,
                a_worker_state_is_kept_in_place_and_survives_appends,
                a_worker_state_read_before_a_purge_is_refused,
                a_purge_of_an_input_deletes_the_state_built_from_it,
                a_resaved_state_leaves_the_inputs_it_no_longer_names,
                worker_states_are_bounded_and_forgotten,
                invalidation_revision_advances_when_only_a_chunk_frame_is_cut,
                invalidation_revision_tracks_late_purge_effects,
                streams_register_once_per_name,
                a_time_scan_returns_exactly_the_window_in_order_across_seq_ties,
                a_range_pages_through_with_its_cursor,
                a_payload_starting_before_the_window_but_reaching_into_it_is_found,
                a_bad_batch_writes_nothing,
                locations_are_kept_per_tier,
                many_payloads_tier_copies_clear_at_once_and_the_unknown_are_skipped,
                provenance_goes_both_ways_and_descendants_are_transitive,
                a_chunk_commit_keeps_every_absorbed_frame_resolvable,
                frame_provenance_rides_with_its_chunk_through_a_rewrite_until_the_chunk_goes,
                a_chunk_committed_without_provenance_has_none_rather_than_an_empty_claim,
                a_repeated_chunk_commit_is_a_duplicate_with_its_sequence_and_a_changed_one_is_refused,
                a_part_joins_its_chunk_absorbs_its_frames_and_moves_it_in_commit_order,
                a_part_that_isnt_its_chunks_next_is_refused_and_changes_nothing,
                a_part_landing_under_a_purge_has_the_purged_frames_cut,
                a_parts_provenance_joins_the_chunks_and_a_silent_side_reads_as_unknown,
                a_chunk_commit_meeting_a_purge_marks_the_purged_frames_cut_and_absorbs_the_rest,
                a_chunk_whose_first_frame_a_purge_hid_commits_under_that_frames_key,
                a_chunk_landing_across_a_purge_window_between_its_frames_has_that_slot_cut,
                a_frame_the_worker_cut_is_left_as_it_was_and_the_chunk_lands_without_it,
                keepers_are_remembered_and_a_purges_acknowledgements_are_recorded,
                a_retried_chunk_commit_under_a_running_purge_is_a_hidden_duplicate,
                a_chunk_that_doesnt_match_its_frames_is_refused,
                a_child_committed_after_its_parent_was_absorbed_is_kept_and_purged_with_it,
                an_identical_header_for_an_absorbed_frame_is_a_repeat_answering_the_chunks_sequence,
                coverage_merges_and_gaps_fill_the_rest,
                a_range_purge_removes_exactly_the_descendants,
                an_entity_purge_removes_its_subject_and_everything_after,
                a_purge_takes_the_purged_spans_out_of_coverage,
                a_range_purge_inside_a_chunk_cuts_frames_and_their_descendants,
                purging_a_whole_chunk_takes_its_absorbed_frames_too,
                an_entity_purge_of_an_absorbed_frame_takes_its_hydrated_copies_and_coverage,
                a_sensor_stream_re_registers_after_a_restart_with_a_fresh_anchor,
                a_payload_cant_be_its_own_parent,
                limits_of_zero_are_refused,
                a_chunk_must_match_the_frames_it_absorbs,
                a_cut_frame_is_marked_and_the_tier_worker_rewrites_the_chunk,
                an_entity_purge_record_keeps_no_trace_of_the_entity,
                a_range_purge_of_an_unknown_stream_is_refused,
                descendants_follow_provenance_not_absorption,
                a_frame_kept_for_its_subjects_still_resolves,
                an_inserts_coverage_mark_merges_on_insert,
                a_stepped_purge_hides_a_budget_at_a_time_and_says_what_it_hid,
                a_payload_derived_from_one_a_purge_hid_is_purged_with_it,
                a_cut_frame_is_in_the_last_steps_delta,
                a_frame_being_cut_stops_being_readable_at_once,
                a_payload_being_purged_takes_no_new_copies,
                overlapping_purges_name_each_row_in_exactly_one_delta,
                a_delta_names_the_copies_a_payload_has_when_it_is_hidden,
                a_hidden_chunk_is_not_rewritten,
                a_chunk_landing_wholly_inside_a_purge_is_doomed_with_its_frames,
                a_child_of_a_payload_the_purge_has_yet_to_reach_is_purged_too,
                a_frame_cut_already_is_not_cut_again,
                purge_steps_refuse_what_they_should,
                a_range_purge_forgets_the_same_frames_before_and_after_compaction,
                a_range_purge_takes_what_is_committed_at_any_point_of_its_scan,
                a_chunk_over_the_frame_limit_is_refused,
                commit_sequences_rise_and_are_never_reused,
                inserting_the_same_payload_again_returns_its_sequence,
                a_query_walks_a_stream_in_commit_order_with_filters_and_a_scan_cap,
                a_purge_by_ids_takes_them_and_what_came_from_them,
                a_windowed_sequence_query_reads_the_window_not_the_history,
                purge_records_list_in_start_order,
                small_payloads_keep_their_bytes_inline,
                a_chunk_takes_a_new_sequence_and_its_frames_leave_the_query,
                a_payload_that_lands_under_a_purge_says_so_and_its_delta_entry_is_late,
                a_payload_landing_where_a_purge_reaches_is_hidden_on_landing,
                inline_bytes_never_move_and_a_purge_report_keeps_none,
                a_chunk_cant_be_placed_inline,
                a_scrub_runs_only_when_due,
                streams_and_coverage_keys_page_in_order,
                readers_run_alongside_a_writer,
                small_tables_keep_what_they_are_given,
                forgotten_hashes_page_lexicographically_without_repeats,
                clock_near_finds_the_points_on_each_side_on_that_boot,
                old_frames_nothing_names_lose_their_rows_and_the_rest_keep_them,
                frames_inside_the_window_or_with_subjects_keep_their_rows,
                a_prune_sweep_goes_a_budget_at_a_time_from_its_cursor,
                a_frame_a_running_purge_holds_is_left_to_the_purge,
                an_age_out_removes_only_what_it_names_and_records_its_spans,
                an_age_out_with_a_gap_records_runs_and_keeps_wider_gaps_apart,
                a_purge_takes_what_depends_on_a_stretch_it_removed_and_what_derives_from_that,
                a_cut_frame_takes_what_depends_on_its_slot,
                an_age_out_leaves_what_depends_on_what_it_aged,
                a_dependency_read_before_a_purge_is_refused,
                an_age_out_keeps_the_worker_state_and_dependencies_read_before_it,
                a_chunk_never_absorbs_a_payload_with_dependencies,
                dependencies_with_one_start_keep_the_longest_reach,
                a_repeat_must_name_the_dependencies_it_was_indexed_with,
                without_a_gap_each_aged_payload_keeps_its_own_span,
                a_wider_gap_joins_runs_left_by_narrower_ones_end_to_end,
                a_completed_age_out_keeps_counting_what_it_removed,
                a_completed_age_out_keeps_its_spans_and_count_but_not_its_items,
                an_age_out_skips_what_was_committed_after_its_cutoff,
                a_child_landing_under_an_age_out_is_not_hidden,
                a_chunk_aged_out_whole_keeps_what_its_frames_led_to,
                recorded_aged_out_spans_merge_and_need_a_known_stream,
            );
        }
    };
    (@tests $make:expr; $($test:ident),* $(,)?) => {
        $(
            #[test]
            fn $test() {
                // A directory for a file-backed catalog, removed when the test ends (after the catalog is dropped).
                let dir = tempfile::tempdir().unwrap();
                super::$test(($make)(dir.path()));
            }
        )*
    };
}

conformance!(memory, |_: &std::path::Path| MemoryCatalog::new());
conformance!(redb, |dir: &std::path::Path| RedbCatalog::open(dir.join("catalog.redb")).unwrap());
