//! Each device's timestamps, turned into host monotonic nanoseconds (docs/CONTINUUM.md, "Timestamps"). Pure
//! functions over what the device API hands back, so they're tested without a device.

/// `V4L2_BUF_FLAG_TIMESTAMP_MASK`: which clock a buffer's timestamp is on.
pub const V4L2_TIMESTAMP_MASK: u32 = 0x0000_e000;
/// `V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC`: `CLOCK_MONOTONIC`.
pub const V4L2_TIMESTAMP_MONOTONIC: u32 = 0x0000_2000;
/// `V4L2_BUF_FLAG_TSTAMP_SRC_MASK`: which moment of the frame the timestamp marks.
pub const V4L2_TSTAMP_SRC_MASK: u32 = 0x0007_0000;
/// `V4L2_BUF_FLAG_TSTAMP_SRC_SOE`: the start of exposure. (End of frame is 0.)
pub const V4L2_TSTAMP_SRC_SOE: u32 = 0x0001_0000;

/// Which moment of a frame its stamp marks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StampSource {
    /// When exposure started (uvcvideo recovers this from USB start-of-frame).
    StartOfExposure,
    /// When the last of the frame arrived, so the stamp is moved back by a frame interval to estimate exposure.
    EndOfFrame,
    /// The device's clock wasn't the monotonic one, so the stamp is when the worker got the frame.
    Arrival,
}

/// A frame placed on the host monotonic clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameStamp {
    /// The best estimate of when the frame was captured, host monotonic nanoseconds.
    pub mono_ns: u64,
    /// The device's own timestamp, raw, nanoseconds (kept in `CaptureInfo` for re-fitting).
    pub device_ns: u64,
    /// What the stamp came from.
    pub source: StampSource,
}

/// Places a V4L2 buffer: its timestamp (`tv_sec`, `tv_usec`) and flags, the frame interval, and when the worker
/// dequeued it (host monotonic). A monotonic start-of-exposure stamp is taken as is; an end-of-frame stamp is moved
/// back by a frame interval; a stamp on any other clock, or none at all, falls back to the arrival time.
#[must_use]
pub fn v4l2_frame(tv_sec: i64, tv_usec: i64, flags: u32, frame_interval_ns: u64, arrival_ns: u64) -> FrameStamp {
    let device_ns = u64::try_from(tv_sec).unwrap_or(0) * 1_000_000_000 + u64::try_from(tv_usec).unwrap_or(0) * 1_000;
    let monotonic = flags & V4L2_TIMESTAMP_MASK == V4L2_TIMESTAMP_MONOTONIC;
    if !monotonic || device_ns == 0 || device_ns > arrival_ns {
        return FrameStamp { mono_ns: arrival_ns, device_ns, source: StampSource::Arrival };
    }
    if flags & V4L2_TSTAMP_SRC_MASK == V4L2_TSTAMP_SRC_SOE {
        FrameStamp { mono_ns: device_ns, device_ns, source: StampSource::StartOfExposure }
    } else {
        FrameStamp { mono_ns: device_ns.saturating_sub(frame_interval_ns), device_ns, source: StampSource::EndOfFrame }
    }
}

/// Nanoseconds for a `CMTime` (an AVFoundation sample buffer's presentation time, on the host clock): `value` over
/// `timescale` seconds. `None` for an invalid time (a timescale of 0, or a negative value).
#[must_use]
pub fn cmtime_ns(value: i64, timescale: i32) -> Option<u64> {
    if timescale <= 0 || value < 0 {
        return None;
    }
    let ns = i128::from(value) * 1_000_000_000 / i128::from(timescale);
    u64::try_from(ns).ok()
}

/// Nanoseconds for an AVFoundation sample buffer's duration (`CMSampleBufferGetDuration`), or zero when it has
/// none: a time that isn't `numeric` (invalid, infinite, or indefinite), zero, negative, or with a bad timescale.
/// Zero keeps the frame an instant; nothing here falls back to a nominal frame rate.
#[must_use]
pub fn sample_duration_ns(value: i64, timescale: i32, numeric: bool) -> u64 {
    if !numeric || value <= 0 {
        return 0;
    }
    cmtime_ns(value, timescale).unwrap_or(0)
}

/// Nanoseconds for host ticks (`mach_absolute_time`, CoreAudio's `mHostTime`), given the timebase's `numer` and
/// `denom` from `mach_timebase_info`.
#[must_use]
pub fn host_ticks_ns(ticks: u64, numer: u32, denom: u32) -> u64 {
    if denom == 0 {
        return ticks;
    }
    u64::try_from(u128::from(ticks) * u128::from(numer) / u128::from(denom)).unwrap_or(u64::MAX)
}

/// Nanoseconds for an ALSA or PipeWire `htimestamp` (a `timespec`).
#[must_use]
pub fn timespec_ns(tv_sec: i64, tv_nsec: i64) -> u64 {
    u64::try_from(tv_sec).unwrap_or(0) * 1_000_000_000 + u64::try_from(tv_nsec).unwrap_or(0)
}

/// When a PipeWire capture buffer's first frame was captured, host monotonic nanoseconds, from the stream's time
/// report (`pw_stream_get_time_n`): `now` is the graph cycle's monotonic time; the newest frame left the device
/// `delay` ticks (at `rate_num / rate_denom` seconds a tick) plus `buffered` frames (in the resampler) before that;
/// and the buffer holds `frames` frames at `sample_rate`.
///
/// This reads `now - delay` as the buffer's newest frame, so the first is a buffer earlier. pipewire's header can be
/// read as `now - delay` naming the first frame instead, which would put every buffer a quantum early here; the
/// worker logs the first report (delay and quantum), and the clap-and-flash run settles which before its offset is
/// recorded. A constant error either way: the fit, the gaps, and batch contiguity don't depend on it.
#[must_use]
pub fn pipewire_capture_ns(
    now_ns: i64,
    delay_ticks: i64,
    rate_num: u32,
    rate_denom: u32,
    buffered_frames: u64,
    frames: u64,
    sample_rate: u32,
) -> u64 {
    let tick_ns = |ticks: i128| {
        if rate_denom == 0 { 0 } else { ticks * i128::from(rate_num) * 1_000_000_000 / i128::from(rate_denom) }
    };
    let frame_ns = |n: u64| if sample_rate == 0 { 0 } else { i128::from(n) * 1_000_000_000 / i128::from(sample_rate) };
    let newest = i128::from(now_ns) - tick_ns(i128::from(delay_ticks)) - frame_ns(buffered_frames);
    let first = newest - frame_ns(frames);
    u64::try_from(first.max(0)).unwrap_or(0)
}

/// [`pipewire_capture_ns`] for every buffer one `process` call took, oldest first: the time report is the newest
/// graph cycle's, so the newest buffer ends where the report says and each earlier one ends where the next begins.
/// A callback that runs late finds several buffers queued; stamping each of them from the report as if it were the
/// newest made them look late, and the audio clock took the lateness for lost samples (Wave 3, 2026-10-30).
#[must_use]
pub fn pipewire_batch_ns(
    now_ns: i64,
    delay_ticks: i64,
    rate_num: u32,
    rate_denom: u32,
    buffered_frames: u64,
    frames: &[u64],
    sample_rate: u32,
) -> Vec<u64> {
    let mut after: u64 = frames.iter().sum();
    frames
        .iter()
        .map(|&f| {
            let at =
                pipewire_capture_ns(now_ns, delay_ticks, rate_num, rate_denom, buffered_frames, after, sample_rate);
            after -= f;
            at
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const MONO_SOE: u32 = V4L2_TIMESTAMP_MONOTONIC | V4L2_TSTAMP_SRC_SOE;
    const FRAME: u64 = 33_333_333;

    #[test]
    fn a_monotonic_start_of_exposure_stamp_is_taken_as_is() {
        let s = v4l2_frame(100, 250_000, MONO_SOE, FRAME, 100_400_000_000);
        assert_eq!(
            s,
            FrameStamp { mono_ns: 100_250_000_000, device_ns: 100_250_000_000, source: StampSource::StartOfExposure }
        );
    }

    #[test]
    fn an_end_of_frame_stamp_moves_back_a_frame() {
        let s = v4l2_frame(100, 250_000, V4L2_TIMESTAMP_MONOTONIC, FRAME, 100_400_000_000);
        assert_eq!((s.mono_ns, s.source), (100_250_000_000 - FRAME, StampSource::EndOfFrame));
    }

    #[test]
    fn a_stamp_on_another_clock_or_from_the_future_falls_back_to_arrival() {
        let copy = 0x0000_4000; // V4L2_BUF_FLAG_TIMESTAMP_COPY
        assert_eq!(v4l2_frame(100, 0, copy, FRAME, 5).source, StampSource::Arrival);
        assert_eq!(v4l2_frame(0, 0, MONO_SOE, FRAME, 5).mono_ns, 5, "no stamp at all");
        let future = v4l2_frame(200, 0, MONO_SOE, FRAME, 100_000_000_000);
        assert_eq!((future.mono_ns, future.source), (100_000_000_000, StampSource::Arrival));
    }

    #[test]
    fn a_cmtime_converts_without_overflow() {
        assert_eq!(cmtime_ns(90_000 * 3_600 * 24, 90_000), Some(86_400_000_000_000));
        assert_eq!(cmtime_ns(i64::MAX, 1_000_000_000), Some(i64::MAX as u64));
        assert_eq!(cmtime_ns(1, 0), None);
        assert_eq!(cmtime_ns(-1, 600), None);
    }

    #[test]
    fn a_sample_duration_counts_only_when_avfoundation_gives_a_positive_numeric_time() {
        // 1/30 s at a 600 timescale, as AVFoundation reports a 30 fps frame.
        assert_eq!(sample_duration_ns(20, 600, true), 33_333_333);
        // kCMTimeInvalid, an indefinite or infinite time, zero, a negative value, and a bad timescale all mean
        // no duration, so the frame stays an instant.
        assert_eq!(sample_duration_ns(0, 0, false), 0);
        assert_eq!(sample_duration_ns(20, 600, false), 0);
        assert_eq!(sample_duration_ns(0, 600, true), 0);
        assert_eq!(sample_duration_ns(-20, 600, true), 0);
        assert_eq!(sample_duration_ns(20, 0, true), 0);
    }

    #[test]
    fn host_ticks_follow_the_timebase() {
        // Apple Silicon's timebase is 125/3: a tick is 41.67 ns.
        assert_eq!(host_ticks_ns(24_000_000, 125, 3), 1_000_000_000);
        assert_eq!(host_ticks_ns(7, 1, 1), 7, "Intel's 1/1");
        assert_eq!(host_ticks_ns(u64::MAX, 125, 3), u64::MAX, "saturates");
    }

    #[test]
    fn a_pipewire_buffer_is_placed_before_the_cycle_by_its_delay_and_length() {
        // A 1024-frame quantum at 48 kHz; the cycle at 10 s; 256 ticks of delay at 1/48000; nothing in the resampler.
        let first = pipewire_capture_ns(10_000_000_000, 256, 1, 48_000, 0, 1_024, 48_000);
        let expected = 10_000_000_000 - 256 * 1_000_000_000 / 48_000 - 1_024 * 1_000_000_000 / 48_000;
        assert!(first.abs_diff(expected) <= 1, "{first} vs {expected}");
        assert_eq!(pipewire_capture_ns(1_000, 0, 1, 0, 0, 0, 0), 1_000, "no rate: the cycle time");
    }

    #[test]
    fn delay_buffering_and_a_graph_clock_at_another_rate_all_count() {
        // A negative delay (the report can run a little ahead) moves the frame later, not earlier.
        let ahead = pipewire_capture_ns(10_000_000_000, -48, 1, 48_000, 0, 480, 48_000);
        assert_eq!(ahead, 10_000_000_000 + 1_000_000 - 10_000_000);
        // Frames waiting in the resampler are older still: 96 frames at 48 kHz is 2 ms.
        let queued = pipewire_capture_ns(10_000_000_000, 0, 1, 48_000, 96, 480, 48_000);
        assert_eq!(queued, 10_000_000_000 - 2_000_000 - 10_000_000);
        // The delay is in the graph clock's ticks (44.1 kHz here), the buffer in the stream's frames (48 kHz).
        let mixed = pipewire_capture_ns(10_000_000_000, 441, 1, 44_100, 0, 480, 48_000);
        assert_eq!(mixed, 10_000_000_000 - 10_000_000 - 10_000_000);
    }

    #[test]
    fn a_timespec_converts() {
        assert_eq!(timespec_ns(3, 500), 3_000_000_500);
    }

    #[test]
    fn a_batch_of_buffers_ends_at_the_report_and_each_follows_the_one_before() {
        let one = pipewire_capture_ns(10_000_000_000, 1536, 1, 48_000, 0, 1024, 48_000);
        let three = pipewire_batch_ns(10_000_000_000, 1536, 1, 48_000, 0, &[1024, 1024, 512], 48_000);
        // the newest (512 frames) ends where a lone 512-frame buffer would; the others are back to back before it
        assert_eq!(three[2], pipewire_capture_ns(10_000_000_000, 1536, 1, 48_000, 0, 512, 48_000));
        assert_eq!(three[1], pipewire_capture_ns(10_000_000_000, 1536, 1, 48_000, 0, 1024 + 512, 48_000));
        assert_eq!(three[0], pipewire_capture_ns(10_000_000_000, 1536, 1, 48_000, 0, 2048 + 512, 48_000));
        assert_eq!(three[1] - three[0], 21_333_333, "a quantum apart, to the nanosecond's rounding");
        assert_eq!(pipewire_batch_ns(10_000_000_000, 1536, 1, 48_000, 0, &[1024], 48_000), vec![one]);
        assert!(pipewire_batch_ns(10_000_000_000, 1536, 1, 48_000, 0, &[], 48_000).is_empty());
    }
}
