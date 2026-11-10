//! Fitting a device's clock to the host's (docs/CONTINUUM.md, "Timestamps").
//!
//! - [`LinearFit`]: least squares over a window of (device position, host time) points, giving an offset and a rate.
//! - [`AudioClock`]: a sample-count clock, anchored to the driver's capture times and re-fitted continuously, with
//!   overruns kept as explicit gaps instead of shifting time.
//! - [`PingFit`]: a serial device's clock (the Teensy's `micros()`), from ping and pong exchanges, keeping the ones
//!   with the smallest round trip.

use std::collections::VecDeque;

/// A least squares line `host = offset + slope * x` over the last `window` points. Points are kept relative to the
/// first, so large timestamps don't lose precision in `f64`.
#[derive(Clone, Debug)]
pub struct LinearFit {
    points: VecDeque<(f64, f64)>,
    window: usize,
    origin: Option<(f64, f64)>,
}

/// A fitted line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Line {
    /// Host time at `x = 0` of the origin frame, as an absolute value.
    x0: f64,
    y0: f64,
    /// Host nanoseconds per unit of `x`.
    pub slope: f64,
    /// Host nanoseconds at `x = x0`.
    intercept: f64,
}

impl Line {
    /// The host time the line puts at `x`.
    #[must_use]
    pub fn at(&self, x: f64) -> f64 {
        self.y0 + self.intercept + self.slope * (x - self.x0)
    }
}

impl LinearFit {
    /// A fit over at most `window` points (at least 2).
    #[must_use]
    pub fn new(window: usize) -> Self {
        Self { points: VecDeque::new(), window: window.max(2), origin: None }
    }

    /// Adds a point, dropping the oldest beyond the window.
    pub fn push(&mut self, x: f64, host: f64) {
        let (ox, oy) = *self.origin.get_or_insert((x, host));
        self.points.push_back((x - ox, host - oy));
        while self.points.len() > self.window {
            self.points.pop_front();
        }
    }

    /// How many points it holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether it holds none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The fitted line, once there are two points with different `x`.
    #[must_use]
    pub fn line(&self) -> Option<Line> {
        let n = self.points.len() as f64;
        if self.points.len() < 2 {
            return None;
        }
        let (mx, my) = self.points.iter().fold((0.0, 0.0), |(a, b), &(x, y)| (a + x / n, b + y / n));
        let (sxx, sxy) =
            self.points.iter().fold((0.0, 0.0), |(a, b), &(x, y)| (a + (x - mx) * (x - mx), b + (x - mx) * (y - my)));
        if sxx == 0.0 {
            return None;
        }
        let slope = sxy / sxx;
        let (ox, oy) = self.origin?;
        Some(Line { x0: ox + mx, y0: oy, slope, intercept: my })
    }

    /// The `x` distance its points cover, first to last (0 with fewer than two).
    #[must_use]
    pub fn span(&self) -> f64 {
        match (self.points.front(), self.points.back()) {
            (Some(first), Some(last)) => last.0 - first.0,
            _ => 0.0,
        }
    }

    /// The point (absolute `x`, host) whose host time is least after its `x` (host minus `x` smallest): for a remote
    /// clock against arrival times, the least-delayed one, which a re-anchor starts from.
    #[must_use]
    pub fn least_lag(&self) -> Option<(f64, f64)> {
        let (ox, oy) = self.origin?;
        self.points.iter().min_by(|a, b| (a.1 - a.0).total_cmp(&(b.1 - b.0))).map(|&(x, y)| (x + ox, y + oy))
    }

    /// The largest distance of a point from `line`, in host units: how far the worst point strays from the fit.
    #[must_use]
    pub fn worst_residual(&self, line: &Line) -> f64 {
        let Some((ox, oy)) = self.origin else { return 0.0 };
        self.points.iter().map(|&(x, y)| (y + oy - line.at(x + ox)).abs()).fold(0.0, f64::max)
    }

    /// Forgets every point (a device restart, a wake).
    pub fn reset(&mut self) {
        self.points.clear();
        self.origin = None;
    }
}

/// A block of audio as the clock places it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockStamp {
    /// Index of the block's first sample in the stream, counting the samples of every gap since the clock started.
    /// A [`AudioClock::reset`] (a wake) carries the count on without counting the sleep: the stream's batch ends
    /// there and its next starts after the sleep, so the catalog shows the hole, and the count is only continuous
    /// between resets.
    pub first_sample: u64,
    /// Host monotonic nanoseconds of its first sample, from the fit.
    pub host_ns: u64,
    /// Samples lost just before this block (an overrun), kept as a gap rather than shifting time; 0 for none.
    pub gap_samples: u64,
}

/// Places audio blocks on the host clock from the driver's capture time for each block (ALSA or PipeWire's
/// htimestamp, CoreAudio's `mHostTime`).
///
/// It counts samples, but anchors the count to the driver's times and re-fits the rate continuously, so
/// the device's drift against the host is followed, not assumed away. When a block's capture time is later than the
/// count says by more than half a block, samples were lost (an overrun): the count jumps by the missing samples (see
/// [`GapUnit`]), and the block says how many, so time never shifts silently.
#[derive(Clone, Debug)]
pub struct AudioClock {
    rate_hz: f64,
    next_sample: u64,
    fit: LinearFit,
    gaps: GapUnit,
}

/// What a gap is counted in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GapUnit {
    /// Samples, from the driver's capture times: for a source whose times come from the driver (PipeWire,
    /// CoreAudio) and whose losses (whole quanta or buffers) don't line up with the stream's blocks.
    #[default]
    Samples,
    /// Whole blocks: for a source whose blocks are its driver's periods, so it can only lose whole blocks, and whose
    /// times are jittery; rounding to blocks keeps the jitter out of the count.
    Blocks,
}

/// How many recent blocks the audio fit uses: at 10 ms blocks, about 20 seconds.
pub const AUDIO_FIT_WINDOW: usize = 2_000;

/// Blocks the fitted line needs before it is trusted over the nominal rate.
pub const MIN_FIT_BLOCKS: usize = 16;

/// The most a sound card's clock is believed to drift from its nominal rate; a fit beyond this is noise, not drift.
pub const MAX_DRIFT_PPM: f64 = 1_000.0;

impl AudioClock {
    /// A clock for a stream nominally at `rate_hz`.
    #[must_use]
    pub fn new(rate_hz: f64) -> Self {
        Self { rate_hz, next_sample: 0, fit: LinearFit::new(AUDIO_FIT_WINDOW), gaps: GapUnit::default() }
    }

    /// Counts gaps in `unit` instead of samples.
    #[must_use]
    pub fn with_gap_unit(mut self, unit: GapUnit) -> Self {
        self.gaps = unit;
        self
    }

    /// Places a block of `samples` (frames, per channel) whose first sample the driver says it captured at
    /// `captured_ns` on the host monotonic clock.
    pub fn block(&mut self, samples: u64, captured_ns: u64) -> BlockStamp {
        let mut gap = 0;
        if let Some(expected) = self.predict(self.next_sample as f64) {
            // Lateness up to half a block (exactly half included) is jitter. Beyond that, samples were lost: as many
            // as the lateness says, or, for a source that loses whole blocks, the nearest whole number of blocks.
            let block_ns = samples as f64 * 1e9 / self.rate_hz;
            let late = captured_ns as f64 - expected;
            if late > block_ns / 2.0 {
                gap = match self.gaps {
                    GapUnit::Samples => (late * self.rate_hz / 1e9).round() as u64,
                    GapUnit::Blocks => (late / block_ns).round() as u64 * samples,
                };
            }
        }
        let first_sample = self.next_sample + gap;
        self.fit.push(first_sample as f64, captured_ns as f64);
        self.next_sample = first_sample + samples;
        let host_ns = self.predict(first_sample as f64).unwrap_or(captured_ns as f64);
        BlockStamp { first_sample, host_ns: host_ns.max(0.0).round() as u64, gap_samples: gap }
    }

    /// Where the clock puts sample `x`: on the fitted line once it is trustworthy (enough blocks, and a rate within
    /// [`MAX_DRIFT_PPM`] of nominal), else at the nominal rate through the mean of the capture times so far, so a
    /// few jittery early blocks can't tilt the line and call jitter a gap.
    fn predict(&self, x: f64) -> Option<f64> {
        if self.fit.len() >= MIN_FIT_BLOCKS
            && let Some(line) = self.fit.line()
        {
            let nominal = 1e9 / self.rate_hz;
            if ((line.slope - nominal) / nominal).abs() * 1e6 <= MAX_DRIFT_PPM {
                return Some(line.at(x));
            }
        }
        let (ox, oy) = self.fit.origin?;
        let n = self.fit.points.len() as f64;
        let ns_per_sample = 1e9 / self.rate_hz;
        let mean_offset = self.fit.points.iter().map(|&(px, py)| py - px * ns_per_sample).sum::<f64>() / n;
        Some(oy + mean_offset + (x - ox) * ns_per_sample)
    }

    /// The device's drift against the host, in parts per million (positive: the device runs fast), once fitted.
    #[must_use]
    pub fn drift_ppm(&self) -> Option<f64> {
        let line = self.fit.line()?;
        let nominal_ns_per_sample = 1e9 / self.rate_hz;
        Some((nominal_ns_per_sample / line.slope - 1.0) * 1e6)
    }

    /// Starts over (a restart, a wake): the next block anchors afresh, and the sample count carries on without the
    /// time between (see [`BlockStamp::first_sample`]).
    pub fn reset(&mut self) {
        self.fit.reset();
    }

    /// Where the clock puts sample `x` now, host monotonic nanoseconds: the end of a batch, so the next batch, which
    /// starts at the same sample, starts at the same time. `None` before the first block.
    #[must_use]
    pub fn host_ns_of(&self, sample: u64) -> Option<u64> {
        self.predict(sample as f64).map(|t| t.max(0.0).round() as u64)
    }
}

/// One ping to a serial device and its pong: the host times it went out and came back, and the device's own time
/// in the pong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ping {
    /// Host monotonic nanoseconds when the ping went out.
    pub sent_ns: u64,
    /// The device's clock in its reply, in its own units: as the device sends it, which may be a counter that
    /// wraps (see [`PingFit::with_wrap_bits`]).
    pub device_ts: u64,
    /// Host monotonic nanoseconds when the reply came back.
    pub received_ns: u64,
}

/// Fits a serial device's clock to the host's from pings. Of every `group` pings it keeps the one with the
/// smallest round trip (the one least delayed on either leg), and places the device's time at that ping's midpoint.
#[derive(Clone, Debug)]
pub struct PingFit {
    group: usize,
    pending: Vec<Ping>,
    fit: LinearFit,
    /// The device counter's width (32 for the Teensy's `micros()`), or 64 for one that doesn't wrap.
    wrap_bits: u32,
    /// The last raw device time seen, and the unwrapped value it stood for.
    last: Option<(u64, u64)>,
}

/// A device time this far from where the fit expects it (host nanoseconds) means the device restarted.
pub const PING_RESTART_NS: f64 = 500_000_000.0;

impl PingFit {
    /// A fit keeping the best of every `group` pings, over the last `window` kept ones.
    #[must_use]
    pub fn new(group: usize, window: usize) -> Self {
        Self { group: group.max(1), pending: Vec::new(), fit: LinearFit::new(window), wrap_bits: 64, last: None }
    }

    /// The device's counter is `bits` wide and wraps (the Teensy's `micros()` is 32 bits: every 71.6 minutes). Its
    /// times are unwrapped into one count before fitting, and [`PingFit::host_ns`] takes the raw value.
    #[must_use]
    pub fn with_wrap_bits(mut self, bits: u32) -> Self {
        self.wrap_bits = bits.clamp(8, 64);
        self
    }

    /// Forgets everything (the device restarted, the host woke).
    pub fn reset(&mut self) {
        self.pending.clear();
        self.fit.reset();
        self.last = None;
    }

    /// The unwrapped value of a raw device time: the one nearest the last unwrapped value.
    fn unwrap(&self, raw: u64) -> u64 {
        let Some((last_raw, last)) = self.last else { return raw };
        if self.wrap_bits >= 64 {
            return raw;
        }
        let mask = (1u64 << self.wrap_bits) - 1;
        let delta = raw.wrapping_sub(last_raw) & mask;
        let half = 1u64 << (self.wrap_bits - 1);
        if delta < half { last + delta } else { last.saturating_sub(mask + 1 - delta) }
    }

    /// Adds a ping. A reply that came back before it went out is ignored. A device time far from where the fit
    /// expects it (more than [`PING_RESTART_NS`]) means the device restarted: the fit starts over from this ping.
    pub fn push(&mut self, ping: Ping) {
        if ping.received_ns < ping.sent_ns {
            return;
        }
        let mid = (ping.sent_ns + (ping.received_ns - ping.sent_ns) / 2) as f64;
        let mut device = self.unwrap(ping.device_ts);
        let restarted = self.last.is_some_and(|(_, last)| device < last)
            || self.fit.line().is_some_and(|l| (l.at(device as f64) - mid).abs() > PING_RESTART_NS);
        if restarted {
            self.reset();
            device = ping.device_ts;
        }
        self.last = Some((ping.device_ts, device));
        let ping = Ping { device_ts: device, ..ping };
        self.pending.push(ping);
        if self.pending.len() >= self.group {
            if let Some(best) = self.pending.iter().min_by_key(|p| p.received_ns - p.sent_ns) {
                let mid = best.sent_ns + (best.received_ns - best.sent_ns) / 2;
                self.fit.push(best.device_ts as f64, mid as f64);
            }
            self.pending.clear();
        }
    }

    /// Host monotonic nanoseconds for a raw device timestamp near the latest pings, once two groups are in.
    #[must_use]
    pub fn host_ns(&self, device_ts: u64) -> Option<u64> {
        let device = self.unwrap(device_ts);
        self.fit.line().map(|l| l.at(device as f64).max(0.0).round() as u64)
    }
}

/// The host time of a control transfer's result (the XVF3800's direction of arrival): the midpoint of the
/// request's round trip.
#[must_use]
pub fn midpoint_ns(sent_ns: u64, received_ns: u64) -> u64 {
    sent_ns + received_ns.saturating_sub(sent_ns) / 2
}

#[cfg(test)]
mod tests {

    #[test]
    fn span_and_worst_residual_describe_the_points_against_their_line() {
        let mut fit = LinearFit::new(10);
        assert_eq!(fit.span(), 0.0);
        for (x, host) in [(1_000.0, 5_000.0), (2_000.0, 6_000.0), (3_000.0, 7_030.0), (4_000.0, 8_000.0)] {
            fit.push(x, host);
        }
        assert_eq!(fit.span(), 3_000.0);
        let line = fit.line().unwrap();
        let worst = fit.worst_residual(&line);
        // The 30-unit bump at x = 3000 is the worst point, and least squares spreads part of it to the others.
        assert!((15.0..=30.0).contains(&worst), "{worst}");
    }
    use super::*;

    #[test]
    fn a_line_through_exact_points_recovers_its_slope_and_offset() {
        let mut f = LinearFit::new(10);
        for x in 0..5 {
            f.push(1e12 + f64::from(x) * 100.0, 5e15 + f64::from(x) * 250.0);
        }
        let l = f.line().unwrap();
        assert!((l.slope - 2.5).abs() < 1e-9);
        assert!((l.at(1e12 + 1000.0) - (5e15 + 2500.0)).abs() < 1e-3);
    }

    #[test]
    fn the_window_keeps_the_last_points_only() {
        let mut f = LinearFit::new(3);
        for x in 0..10 {
            f.push(f64::from(x), f64::from(x) * if x < 7 { 1.0 } else { 10.0 });
        }
        assert_eq!(f.len(), 3);
        assert!((f.line().unwrap().slope - 10.0).abs() < 1e-9, "only the last three, at slope 10");
    }

    #[test]
    fn audio_drift_is_fitted_from_capture_times() {
        // A 48 kHz device that runs 50 ppm fast: each 480-sample block arrives a little sooner than nominal.
        let mut c = AudioClock::new(48_000.0);
        let block_ns = 10_000_000.0 / (1.0 + 50e-6);
        for i in 0..1_000u32 {
            let s = c.block(480, 1_000_000_000 + (f64::from(i) * block_ns) as u64);
            assert_eq!(s.gap_samples, 0);
        }
        let ppm = c.drift_ppm().unwrap();
        assert!((ppm - 50.0).abs() < 0.5, "fitted {ppm} ppm");
    }

    #[test]
    fn an_overrun_is_an_explicit_gap_and_time_does_not_shift() {
        let mut c = AudioClock::new(48_000.0);
        let mut t = 1_000_000_000u64;
        for _ in 0..100 {
            c.block(480, t);
            t += 10_000_000;
        }
        // Three blocks are lost: the next block arrives 30 ms later than the count says.
        t += 30_000_000;
        let s = c.block(480, t);
        assert_eq!(s.gap_samples, 1_440);
        assert_eq!(s.first_sample, 100 * 480 + 1_440);
        assert!(s.host_ns.abs_diff(t) < 100_000, "the block is placed where it was captured");
    }

    /// Feeds 100 clean 480-sample blocks at 48 kHz, loses `lost` samples, and returns the next block's stamp. Times
    /// are truncated to whole nanoseconds, as a driver reports them.
    fn after_a_loss_of(lost: u64) -> BlockStamp {
        let mut c = AudioClock::new(48_000.0);
        let at = |sample: u64| 1_000_000_000 + (sample as f64 * 1e9 / 48_000.0) as u64;
        for i in 0..100 {
            c.block(480, at(i * 480));
        }
        c.block(480, at(100 * 480 + lost))
    }

    #[test]
    fn a_loss_of_exactly_half_a_block_is_jitter() {
        assert_eq!(after_a_loss_of(240).gap_samples, 0);
    }

    #[test]
    fn a_loss_one_sample_over_half_a_block_is_counted_exactly() {
        let s = after_a_loss_of(241);
        assert_eq!(s.gap_samples, 241);
        assert_eq!(s.first_sample, 100 * 480 + 241);
    }

    #[test]
    fn jitter_under_half_a_block_is_not_a_gap() {
        let mut c = AudioClock::new(16_000.0);
        let mut t = 0u64;
        for i in 0..200u64 {
            let jitter = if i % 3 == 0 { 4_000_000 } else { 0 }; // 4 ms of a 10 ms block
            assert_eq!(c.block(160, t + jitter).gap_samples, 0);
            t += 10_000_000;
        }
    }

    #[test]
    fn a_ping_fit_keeps_the_fastest_round_trip_of_each_group() {
        // The device's microsecond clock runs from 0 at host 1 s; one ping in each group of four is quick.
        let mut f = PingFit::new(4, 50);
        for g in 0..20u64 {
            for i in 0..4u64 {
                let sent = 1_000_000_000 + g * 100_000_000 + i * 1_000_000;
                let (out, back) = if i == 2 { (50_000, 50_000) } else { (900_000, 100_000) };
                let device_us = (sent + out - 1_000_000_000) / 1_000;
                f.push(Ping { sent_ns: sent, device_ts: device_us, received_ns: sent + out + back });
            }
        }
        // The kept pings had symmetric 50 us legs, so their midpoint is the device's moment, to the microsecond.
        let host = f.host_ns(5_000_000).unwrap();
        assert!(host.abs_diff(6_000_000_000) < 2_000, "{host}");
    }

    #[test]
    fn a_control_transfer_is_stamped_at_its_midpoint() {
        assert_eq!(midpoint_ns(100, 300), 200);
        assert_eq!(midpoint_ns(300, 100), 300, "a backwards clock gives the send time");
    }

    /// Pings every 100 ms to a device counting microseconds from `start_us`, 50 ppm fast, with a 1 ms round trip.
    fn teensy_pings(fit: &mut PingFit, start_us: u64, host_start_ns: u64, n: u64) {
        for i in 0..n {
            let sent = host_start_ns + i * 100_000_000;
            let device_us = start_us + (i as f64 * 100_000.0 * 1.000_05) as u64;
            fit.push(Ping { sent_ns: sent, device_ts: device_us & 0xffff_ffff, received_ns: sent + 1_000_000 });
        }
    }

    #[test]
    fn a_32_bit_device_counter_is_unwrapped_across_its_wrap() {
        let mut f = PingFit::new(1, 50).with_wrap_bits(32);
        // Starts 3 s before micros() wraps and runs 10 s past it.
        let start = (1u64 << 32) - 3_000_000;
        teensy_pings(&mut f, start, 1_000_000_000, 130);
        let device_now = (start + (129.0 * 100_000.0 * 1.000_05) as u64) & 0xffff_ffff;
        let host = f.host_ns(device_now).unwrap();
        let expected = 1_000_000_000 + 129 * 100_000_000 + 500_000;
        assert!(host.abs_diff(expected) < 50_000, "{host} vs {expected}");
    }

    #[test]
    fn a_device_restart_starts_the_fit_over() {
        let mut f = PingFit::new(1, 50).with_wrap_bits(32);
        teensy_pings(&mut f, 900_000_000, 1_000_000_000, 40);
        // It reboots: micros() starts from 0 again, 4 s of host time later.
        teensy_pings(&mut f, 0, 5_000_000_000, 10);
        let host = f.host_ns((9.0 * 100_000.0 * 1.000_05) as u64).unwrap();
        let expected = 5_000_000_000 + 9 * 100_000_000 + 500_000;
        assert!(host.abs_diff(expected) < 50_000, "{host} vs {expected}");
        f.reset();
        assert_eq!(f.host_ns(0), None);
    }
}
