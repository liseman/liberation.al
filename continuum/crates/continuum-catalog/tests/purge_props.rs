//! A purge over any step boundaries, with inserts landing between steps: each row is named in exactly one delta, a
//! named row is never read again, the deltas name exactly what the report removed, and `done` never goes back.

use std::collections::BTreeSet;

use continuum_catalog::*;
use continuum_core::{DomainLayer, HeaderBuf, HeaderSpec, Interval, PayloadFormat, StreamId, Uuid};
use proptest::prelude::*;

fn id(n: u32) -> Uuid {
    let mut b = [0; 16];
    b[12..].copy_from_slice(&n.to_be_bytes());
    Uuid(b)
}

fn info(name: &str) -> StreamInfo {
    StreamInfo {
        name: name.into(),
        kind: StreamKind::Sensor,
        layer: DomainLayer::Layer1Raw,
        format: PayloadFormat::RawBytes,
        shape_name: String::new(),
        shape_version: 0,
        rate_hz: None,
        anchor: None,
    }
}

fn payload(n: u32, stream: StreamId, t: u64, seq: u32, parents: &[u32], subjects: &[u32]) -> NewPayload {
    let mut spec = HeaderSpec::raw(id(n), stream, t, t);
    spec.seq = seq;
    spec.parents = parents.iter().map(|&p| id(p)).collect();
    spec.subjects = subjects.iter().map(|&s| id(s)).collect();
    NewPayload { header: HeaderBuf::build(&spec).unwrap(), location: None, coverage: None }
}

/// Frames 1..=20 on cam0 at 10 n; crops 100 + n from each (subject 900 on even n); embeddings 200 + n from each crop.
fn graph<C: Catalog>(c: &C) -> (StreamId, StreamId) {
    let cam = c.register_stream(info("cam0")).unwrap();
    let d = c.register_stream(info("derived")).unwrap();
    let mut batch = Vec::new();
    for n in 1..=20u32 {
        let t = u64::from(n) * 10;
        let subjects: &[u32] = if n % 2 == 0 { &[900] } else { &[] };
        batch.push(payload(n, cam, t, 0, &[], &[]));
        batch.push(payload(100 + n, d, t, 0, &[n], subjects));
        batch.push(payload(200 + n, d, t, 1, &[100 + n], &[]));
    }
    c.insert(batch).unwrap();
    (cam, d)
}

/// Runs one purge over the graph with these step budgets, inserting one new payload before each step whose
/// parent is `parents[k]` (any id in the graph, hidden or not), and checks the invariants.
fn check<C: Catalog>(
    c: &C,
    entity: bool,
    budgets: &[usize],
    parents: &[u32],
) -> std::result::Result<(), TestCaseError> {
    let (cam, d) = graph(c);
    let purge = id(5_000);
    let selector = if entity {
        PurgeSelector::Entity(id(900))
    } else {
        PurgeSelector::Range { span: Interval::new(50, 150).unwrap(), streams: Some(vec![cam]) }
    };
    c.start_purge(selector, "prop", "", purge, 1).unwrap();
    let mut named: Vec<Uuid> = Vec::new();
    let mut done = 0;
    let mut next = 1_000u32;
    for k in 0.. {
        if let Some(&p) = parents.get(k) {
            // A late worker result, if its parent is still known (a removed parent is refused, as it should be).
            let _ = c.insert(vec![payload(next, d, 1_000 + u64::from(next), 0, &[p], &[])]);
            next += 1;
        }
        let budget = budgets.get(k).copied().unwrap_or(3).max(1);
        let (delta, report) = match c.purge_step(purge, budget).unwrap() {
            PurgeProgress::Running { delta, done: now, .. } => {
                prop_assert!(now >= done, "done went back from {done} to {now}");
                done = now;
                (delta, None)
            }
            PurgeProgress::Removed { report, delta } => (delta, Some(report)),
        };
        named.extend(delta.doomed.iter().map(|x| x.id));
        for x in &named {
            prop_assert!(c.get(*x).unwrap().is_none(), "{x} was named but is still read");
        }
        if let Some(report) = report {
            let unique: BTreeSet<Uuid> = named.iter().copied().collect();
            prop_assert_eq!(unique.len(), named.len(), "a row was named twice");
            let removed: BTreeSet<Uuid> = report.removed.iter().map(|r| r.id).collect();
            prop_assert_eq!(unique, removed, "the deltas and the report disagree");
            return Ok(());
        }
        prop_assert!(k < 10_000, "the purge never finished");
    }
    Ok(())
}

fn ids_in_graph() -> impl Strategy<Value = u32> {
    prop_oneof![1u32..=20, 101u32..=120, 201u32..=220]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn a_stepped_purge_keeps_its_promises_in_memory(
        entity in any::<bool>(),
        budgets in prop::collection::vec(1usize..6, 0..40),
        parents in prop::collection::vec(ids_in_graph(), 0..20),
    ) {
        check(&MemoryCatalog::new(), entity, &budgets, &parents)?;
    }
}

proptest! {
    // Each case opens a file and commits every step with an fsync, so fewer cases.
    #![proptest_config(ProptestConfig::with_cases(12))]

    #[test]
    fn a_stepped_purge_keeps_its_promises_in_redb(
        entity in any::<bool>(),
        budgets in prop::collection::vec(1usize..6, 0..40),
        parents in prop::collection::vec(ids_in_graph(), 0..20),
    ) {
        let dir = tempfile::tempdir().unwrap();
        check(&RedbCatalog::open(dir.path().join("catalog.redb")).unwrap(), entity, &budgets, &parents)?;
    }
}
