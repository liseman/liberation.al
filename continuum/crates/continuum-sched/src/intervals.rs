//! Interval set arithmetic on `continuum_core::Interval` (half-open, never empty): merging, subtraction, and
//! clipping. The fake catalog uses it, and so will the planner (a quote is a range minus what exists).
//!
//! Every function here takes and returns intervals sorted by start and pairwise disjoint (a "set"), except where
//! the doc says the input may be arbitrary. Nothing here adds to a `u64`, so `u64::MAX` ends are safe.

use continuum_core::Interval;

/// Sorts `intervals` and merges the ones that overlap or touch, so the result is a set.
#[must_use]
pub fn merge(mut intervals: Vec<Interval>) -> Vec<Interval> {
    intervals.sort_by_key(|i| (i.start_ns(), i.end_ns()));
    let mut out: Vec<Interval> = Vec::with_capacity(intervals.len());
    for i in intervals {
        match out.last_mut() {
            Some(last) if last.end_ns() >= i.start_ns() => {
                if i.end_ns() > last.end_ns() {
                    *last = Interval::new(last.start_ns(), i.end_ns()).expect("start < end: end grew");
                }
            }
            _ => out.push(i),
        }
    }
    out
}

/// `window` minus the set `taken`: the parts of `window` no interval in `taken` covers, in time order.
#[must_use]
pub fn subtract(window: Interval, taken: &[Interval]) -> Vec<Interval> {
    let mut out = Vec::new();
    let mut cursor = window.start_ns();
    for t in taken {
        if t.end_ns() <= cursor {
            continue;
        }
        if t.start_ns() >= window.end_ns() {
            break;
        }
        if t.start_ns() > cursor {
            out.push(Interval::new(cursor, t.start_ns()).expect("cursor < start"));
        }
        cursor = cursor.max(t.end_ns());
    }
    if cursor < window.end_ns() {
        out.push(Interval::new(cursor, window.end_ns()).expect("cursor < end"));
    }
    out
}

/// The part of `i` inside `window`, or `None` if they don't overlap.
#[must_use]
pub fn clip(i: Interval, window: Interval) -> Option<Interval> {
    Interval::new(i.start_ns().max(window.start_ns()), i.end_ns().min(window.end_ns()))
}

/// The set `set` clipped to `window`.
#[must_use]
pub fn clip_all(set: &[Interval], window: Interval) -> Vec<Interval> {
    set.iter().filter_map(|&i| clip(i, window)).collect()
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn iv(start: u64, end: u64) -> Interval {
        Interval::new(start, end).expect("test interval")
    }

    fn is_set(v: &[Interval]) -> bool {
        v.windows(2).all(|w| w[0].end_ns() < w[1].start_ns())
    }

    fn covers(set: &[Interval], t: u64) -> bool {
        set.iter().any(|i| i.contains(t))
    }

    #[test]
    fn merge_joins_touching_and_overlapping() {
        assert_eq!(merge(vec![iv(10, 20), iv(0, 10), iv(15, 30), iv(40, 50)]), vec![iv(0, 30), iv(40, 50)]);
        assert_eq!(merge(vec![iv(0, 50), iv(10, 20)]), vec![iv(0, 50)]);
        assert_eq!(merge(Vec::new()), Vec::<Interval>::new());
    }

    #[test]
    fn subtract_leaves_the_uncovered_parts() {
        let taken = merge(vec![iv(10, 20), iv(20, 25), iv(40, 50)]);
        assert_eq!(subtract(iv(0, 60), &taken), vec![iv(0, 10), iv(25, 40), iv(50, 60)]);
        assert_eq!(subtract(iv(12, 18), &taken), Vec::<Interval>::new());
        assert_eq!(subtract(iv(45, 100), &taken), vec![iv(50, 100)]);
        assert_eq!(subtract(iv(0, 60), &[]), vec![iv(0, 60)]);
        assert_eq!(subtract(iv(0, u64::MAX), &[iv(0, u64::MAX)]), Vec::<Interval>::new());
    }

    #[test]
    fn clip_is_the_overlap() {
        assert_eq!(clip(iv(0, 10), iv(5, 20)), Some(iv(5, 10)));
        assert_eq!(clip(iv(0, 10), iv(10, 20)), None);
        assert_eq!(clip(iv(0, 100), iv(10, 20)), Some(iv(10, 20)));
    }

    fn small_interval() -> impl Strategy<Value = Interval> {
        (0u64..200, 1u64..40).prop_map(|(s, len)| iv(s, s + len))
    }

    fn small_intervals() -> impl Strategy<Value = Vec<Interval>> {
        prop::collection::vec(small_interval(), 0..12)
    }

    proptest! {
        #[test]
        fn merge_is_a_set_with_the_same_points(input in small_intervals()) {
            let m = merge(input.clone());
            prop_assert!(is_set(&m));
            for t in 0..260u64 {
                prop_assert_eq!(covers(&m, t), covers(&input, t), "point {}", t);
            }
            prop_assert_eq!(merge(m.clone()), m);
        }

        #[test]
        fn merge_ignores_input_order(input in small_intervals(), seed in any::<u64>()) {
            let mut shuffled = input.clone();
            // A deterministic shuffle from the seed: swap each element with one before it.
            let mut s = seed;
            for i in (1..shuffled.len()).rev() {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                let j = usize::try_from(s % (i as u64 + 1)).expect("fits");
                shuffled.swap(i, j);
            }
            prop_assert_eq!(merge(shuffled), merge(input));
        }

        #[test]
        fn subtract_tiles_the_window_with_the_taken_part(taken in small_intervals(), window in small_interval()) {
            let taken = merge(taken);
            let gaps = subtract(window, &taken);
            prop_assert!(is_set(&gaps));
            for g in &gaps {
                prop_assert!(window.start_ns() <= g.start_ns() && g.end_ns() <= window.end_ns());
            }
            for t in window.start_ns()..window.end_ns() {
                prop_assert_eq!(covers(&gaps, t), !covers(&taken, t), "point {}", t);
            }
        }
    }
}
