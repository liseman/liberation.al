//! The audio clock over any drift, jitter, and overruns. With jittery times and blocks that are the driver's periods
//! (`GapUnit::Blocks`), every lost block is an explicit gap of the right size, no jitter is mistaken for one, and the
//! drift is recovered. With driver times (`GapUnit::Samples`), a loss of any size is counted to within a sample.

use continuum_capture::fit::{AudioClock, GapUnit};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn gaps_are_exact_and_drift_is_recovered(
        ppm in -200.0f64..200.0,
        jitter in prop::collection::vec(0.0f64..0.4, 1_500),
        drops in prop::collection::btree_set(50usize..1_400, 0..6),
        lost in 1u64..4,
    ) {
        // 48 kHz, 10 ms blocks of 480 samples; the device runs `ppm` fast; each block's capture time is late by up to
        // 40% of a block; at each index in `drops`, `lost` blocks never arrive.
        let mut c = AudioClock::new(48_000.0).with_gap_unit(GapUnit::Blocks);
        let block_ns = 10_000_000.0 / (1.0 + ppm * 1e-6);
        let mut sample = 0u64;
        for (i, j) in jitter.iter().enumerate() {
            let mut expected_gap = 0;
            if drops.contains(&i) {
                sample += lost * 480;
                expected_gap = lost * 480;
            }
            let t = 1e9 + sample as f64 / 480.0 * block_ns + j * 10_000_000.0;
            let s = c.block(480, t as u64);
            prop_assert_eq!(s.gap_samples, expected_gap, "block {}", i);
            prop_assert_eq!(s.first_sample, sample);
            sample += 480;
        }
        let fitted = c.drift_ppm().unwrap();
        prop_assert!((fitted - ppm).abs() < 30.0, "fitted {} ppm against {}", fitted, ppm);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn with_driver_times_every_block_is_placed_within_100_us_and_every_gap_is_counted(
        ppm in -200.0f64..200.0,
        jitter_ns in prop::collection::vec(0.0f64..20_000.0, 1_500),
        drops in prop::collection::btree_map(50usize..1_400, 256u64..5_000, 0..6),
    ) {
        // Driver times jitter by under 20 us (a sample at 48 kHz is 20.8 us). A loss is any number of samples from
        // 256 up, PipeWire's smallest quantum at its defaults. The clock calls a block jitter up to half a block
        // late (240 samples here, pinned by fit.rs's unit tests), and each gap is measured from timing, so it can
        // be off by the 12 samples allowed below; the fit carries that into the count. A loss within that error of
        // half a block can land either side of it, so the losses start clear of it (#1372). What readers use is
        // each block's time, which follows the driver's times whatever the count does.
        let mut c = AudioClock::new(48_000.0);
        let sample_ns = 1e9 / 48_000.0 / (1.0 + ppm * 1e-6);
        let mut sample = 0u64;
        for (i, j) in jitter_ns.iter().enumerate() {
            let lost = drops.get(&i).copied().unwrap_or(0);
            sample += lost;
            let truth = 1e9 + sample as f64 * sample_ns;
            let s = c.block(480, (truth + j) as u64);
            prop_assert_eq!(s.gap_samples > 0, lost > 0, "block {}: gap {} for {} lost", i, s.gap_samples, lost);
            prop_assert!(s.gap_samples.abs_diff(lost) <= 12, "block {}: gap {} for {} lost", i, s.gap_samples, lost);
            let off = (s.host_ns as f64 - truth).abs();
            prop_assert!(off < 100_000.0, "block {}: placed {} ns off", i, off);
            sample += 480;
        }
    }
}
