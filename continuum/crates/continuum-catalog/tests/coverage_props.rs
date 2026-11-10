//! Coverage over any insert order: a set of disjoint, merged intervals, and gaps plus coverage tile the window.

use continuum_catalog::CoverageSet;
use continuum_core::Interval;
use proptest::prelude::*;

fn interval() -> impl Strategy<Value = Interval> {
    (0u64..1_000, 1u64..200).prop_map(|(s, len)| Interval::new(s, s + len).unwrap())
}

/// The covered nanoseconds as a bitmap over [0, 1200), computed directly from the inserts.
fn bitmap(spans: &[Interval]) -> Vec<bool> {
    let mut bits = vec![false; 1_200];
    for s in spans {
        bits[s.start_ns() as usize..s.end_ns() as usize].fill(true);
    }
    bits
}

proptest! {
    #[test]
    fn coverage_stays_disjoint_and_merged_under_any_insert_order(spans in prop::collection::vec(interval(), 0..40)) {
        let mut c = CoverageSet::new();
        for &s in &spans {
            c.insert(s);
        }
        let stored: Vec<Interval> = c.iter().collect();
        for pair in stored.windows(2) {
            prop_assert!(pair[0].end_ns() < pair[1].start_ns(), "{:?} and {:?} touch or overlap", pair[0], pair[1]);
        }
        let mut covered = vec![false; 1_200];
        for s in &stored {
            covered[s.start_ns() as usize..s.end_ns() as usize].fill(true);
        }
        prop_assert_eq!(covered, bitmap(&spans));
    }

    #[test]
    fn gaps_and_coverage_tile_the_window(
        spans in prop::collection::vec(interval(), 0..40),
        removed in prop::collection::vec(interval(), 0..5),
        window in interval(),
    ) {
        let mut c = CoverageSet::new();
        for &s in &spans {
            c.insert(s);
        }
        for &r in &removed {
            c.remove(r);
        }
        let mut pieces: Vec<Interval> = c.within(window);
        pieces.extend(c.gaps(window, usize::MAX));
        pieces.sort();
        prop_assert_eq!(pieces.first().map(|p| p.start_ns()), Some(window.start_ns()));
        prop_assert_eq!(pieces.last().map(|p| p.end_ns()), Some(window.end_ns()));
        for pair in pieces.windows(2) {
            prop_assert_eq!(pair[0].end_ns(), pair[1].start_ns(), "a hole or an overlap between pieces");
        }
        let mut expected = bitmap(&spans);
        for r in &removed {
            expected[r.start_ns() as usize..r.end_ns() as usize].fill(false);
        }
        for g in c.gaps(window, usize::MAX) {
            prop_assert!((g.start_ns()..g.end_ns()).all(|t| !expected[t as usize]), "gap {g:?} holds covered time");
        }
        for w in c.within(window) {
            prop_assert!((w.start_ns()..w.end_ns()).all(|t| expected[t as usize]), "{w:?} holds removed time");
        }
    }
}
