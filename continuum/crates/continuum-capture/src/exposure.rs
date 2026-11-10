//! Steering a camera's exposure to the room (#535), the way the listen worker's auto gain steers a microphone.
//!
//! [`AutoExposure`] is pure: no clock or device of its own. About once a second the capture worker measures a frame's
//! brightness ([`luma`]: its mean 8-bit luma and the share of it clipped at white) and asks [`AutoExposure::step`]
//! what exposure to set; the device side ([`Exposure`]) sets it.
//!
//! - **Toward a target, slowly.** Brightness grows about in proportion to exposure, so the next exposure is the
//!   current one scaled by target over measured, but never by more than [`Settings::max_step`] at once, never more
//!   often than [`Settings::interval_ns`], and not at all within [`Settings::deadband`] of the target: faces want a
//!   steady picture, and a camera's own processing lags a change by a few frames.
//!   When rounding would leave the control unchanged, move one legal device step: zero and small brightness
//!   values must still be able to leave a dark or clipped picture. Signed controls scale the magnitude of their
//!   change, so brightening always increases the control rather than making a negative value more negative.
//! - **Clipping first.** A frame with more than [`Settings::clip_share`] of its pixels at white steps down at once,
//!   whatever the mean says: a window behind someone shouldn't hold the room's exposure up. The step grows with how
//!   far over the share it is, from [`Settings::fine_step`] just over it to `max_step` at twice it.
//! - **Not back into the clip.** The value that clipped becomes a ceiling, and the exposure doesn't step up past
//!   `fine_step` under it, even when the mean is below the target: a lamp in a 360 view can clip before the room
//!   reaches the target, and without the ceiling the clip rule and the mean rule took turns every interval (#3149).
//!   The ceiling relaxes by `fine_step` every [`Settings::relax_ns`], to find out whether the light changed, and
//!   every interval while the frame clips under a quarter of `clip_share` (the lamp went off).
//! - **Within the device's range, and the frame rate held.** Every value is clamped to the control's minimum,
//!   maximum, and step, and never past the frame period: in dim light the camera's gain rises instead of the shutter
//!   slowing (`hold_fps`).

/// A control's range, as the driver reports it (`VIDIOC_QUERYCTRL`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range {
    /// The smallest value.
    pub min: i64,
    /// The largest.
    pub max: i64,
    /// The step between values (1 when the driver says 0).
    pub step: i64,
}

impl Range {
    /// `value` within the range, on a step.
    #[must_use]
    pub fn clamp(&self, value: i64) -> i64 {
        let step = self.step.max(1);
        let v = value.clamp(self.min, self.max);
        let snapped = self.min + (v - self.min) / step * step;
        snapped.clamp(self.min, self.max)
    }
}

/// How the controller steers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    /// The mean luma (0 to 255) it steers toward.
    pub target: f32,
    /// No change while the measured luma is within this of the target.
    pub deadband: f32,
    /// The largest proportional change at once; quantization may require one larger device step.
    pub max_step: f32,
    /// The least time between changes, nanoseconds.
    pub interval_ns: u64,
    /// The share of clipped pixels that steps the exposure down at once.
    pub clip_share: f32,
    /// The smallest proportional step a clipped frame takes, and how far under the value that last clipped the
    /// exposure stays.
    pub fine_step: f32,
    /// How long the clip ceiling holds before it relaxes by one `fine_step`, nanoseconds.
    pub relax_ns: u64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            target: 110.0,
            deadband: 12.0,
            max_step: 0.2,
            interval_ns: 2_000_000_000,
            clip_share: 0.05,
            fine_step: 0.05,
            relax_ns: 1_200_000_000_000,
        }
    }
}

/// A frame's brightness: its mean 8-bit luma and the share of its pixels at or above 250.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Luma {
    /// Mean luma, 0 to 255.
    pub mean: f32,
    /// The share of pixels clipped at white, 0 to 1.
    pub clipped: f32,
}

/// The brightness of a luma plane (every `stride`th byte of `plane`, so a packed format's Y samples can be read in
/// place). None for an empty one.
#[must_use]
pub fn luma(plane: &[u8], stride: usize) -> Option<Luma> {
    let stride = stride.max(1);
    let (mut sum, mut n, mut white) = (0u64, 0u64, 0u64);
    for &y in plane.iter().step_by(stride) {
        sum += u64::from(y);
        n += 1;
        white += u64::from(y >= 250);
    }
    #[allow(clippy::cast_precision_loss)]
    (n > 0).then(|| Luma { mean: sum as f32 / n as f32, clipped: white as f32 / n as f32 })
}

/// A frame's luma as a plane of `width` by `height` bytes: an MJPEG frame decoded (luma only), the Y samples of a
/// packed 4:2:2 frame (UYVY, YUYV) read at their row stride, or an NV12 frame's first plane. None for a frame that
/// doesn't decode or a shape this doesn't know.
#[must_use]
pub fn luma_plane(shape: &str, bytes: &[u8], width: u32, height: u32, stride: u32) -> Option<(Vec<u8>, usize, usize)> {
    let (w, h, stride) = (width as usize, height as usize, stride as usize);
    match shape {
        "MJPEG" => {
            use zune_jpeg::JpegDecoder;
            use zune_jpeg::zune_core::bytestream::ZCursor;
            use zune_jpeg::zune_core::colorspace::ColorSpace;
            use zune_jpeg::zune_core::options::DecoderOptions;
            let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::Luma);
            let mut d = JpegDecoder::new_with_options(ZCursor::new(bytes), options);
            let plane = d.decode().ok()?;
            let info = d.info()?;
            let (w, h) = (usize::from(info.width), usize::from(info.height));
            (plane.len() >= w * h).then_some((plane, w, h))
        }
        "UYVY" | "2vuy" | "YUYV" | "YUY2" => {
            let first = usize::from(matches!(shape, "UYVY" | "2vuy"));
            let stride = if stride == 0 { w * 2 } else { stride };
            if bytes.len() < stride * h.saturating_sub(1) + w * 2 {
                return None;
            }
            let plane = (0..h).flat_map(|y| (0..w).map(move |x| bytes[y * stride + 2 * x + first])).collect();
            Some((plane, w, h))
        }
        "NV12" => {
            let stride = if stride == 0 { w } else { stride };
            if bytes.len() < stride * h.saturating_sub(1) + w {
                return None;
            }
            Some(((0..h).flat_map(|y| bytes[y * stride..y * stride + w].iter().copied()).collect(), w, h))
        }
        _ => None,
    }
}

/// The brightness inside `boxes` (each `[x1, y1, x2, y2]`, normalized 0 to 1) of a `width` by `height` luma plane:
/// the faces in view, when steering for them. None when no box covers a pixel.
#[must_use]
pub fn luma_in(plane: &[u8], width: usize, height: usize, boxes: &[[f32; 4]]) -> Option<Luma> {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
    let px = |v: f32, n: usize| ((v.clamp(0.0, 1.0) * n as f32) as usize).min(n);
    let mut picked = Vec::new();
    for b in boxes {
        let (x1, x2, y1, y2) = (px(b[0], width), px(b[2], width), px(b[1], height), px(b[3], height));
        for y in y1..y2 {
            picked.extend_from_slice(&plane[y * width + x1..y * width + x2]);
        }
    }
    luma(&picked, 1)
}

/// Setting one camera's exposure (V4L2's `exposure_time_absolute`, AVFoundation's exposure duration).
pub trait Exposure {
    /// The exposure control's range.
    fn range(&self) -> Range;
    /// The exposure now.
    fn get(&mut self) -> Option<i64>;
    /// Sets it, or why the camera refused (with its errno).
    ///
    /// # Errors
    ///
    /// The camera's refusal.
    fn set(&mut self, value: i64) -> Result<(), String>;
}

/// Decides when and how far to move one camera's exposure (see the module doc).
#[derive(Clone, Debug)]
pub struct AutoExposure {
    range: Range,
    /// The longest exposure that keeps the frame rate, in the control's units.
    cap: i64,
    settings: Settings,
    last_change_ns: Option<u64>,
    /// The value that last clipped (or the ceiling relaxed from it) and when it was set or last relaxed.
    ceiling: Option<(i64, u64)>,
    /// Why the last [`AutoExposure::step`] did what it did, for the status line.
    last: &'static str,
}

impl AutoExposure {
    /// A controller for an exposure control with `range`, whose units are `unit_ns` nanoseconds (100 µs for V4L2's
    /// `exposure_time_absolute`), on a camera delivering a frame every `frame_interval_ns`.
    #[must_use]
    pub fn new(range: Range, unit_ns: u64, frame_interval_ns: u64, settings: Settings) -> Self {
        let cap = range.clamp(i64::try_from(frame_interval_ns / unit_ns.max(1)).unwrap_or(i64::MAX));
        Self { range, cap, settings, last_change_ns: None, ceiling: None, last: "not measured yet" }
    }

    /// A controller for a control with no frame-rate cap (a camera's brightness, when its exposure can't be set).
    #[must_use]
    pub fn uncapped(range: Range, settings: Settings) -> Self {
        let cap = range.clamp(range.max);
        Self { range, cap, settings, last_change_ns: None, ceiling: None, last: "not measured yet" }
    }

    /// The last change didn't take (the camera refused it, or kept its value): the status says so instead of
    /// "changed".
    pub fn refused(&mut self) {
        self.last = "refused by the camera";
    }

    /// The longest exposure it will set (the frame period, within the range).
    #[must_use]
    pub fn cap(&self) -> i64 {
        self.cap
    }

    /// Why the last [`AutoExposure::step`] did what it did: "changed", "waiting" (too soon after a change), "in the
    /// deadband", "under the clip ceiling" (the value that last clipped), "at the cap" (the frame period), or "at a
    /// bound" (the control's range).
    #[must_use]
    pub fn last(&self) -> &'static str {
        self.last
    }

    /// The exposure to set now, given the current one and the frame's brightness, or None to leave it.
    pub fn step(&mut self, now_ns: u64, current: i64, luma: Luma) -> Option<i64> {
        if self.last_change_ns.is_some_and(|t| now_ns.saturating_sub(t) < self.settings.interval_ns) {
            self.last = "waiting";
            return None;
        }
        let s = self.settings;
        #[allow(clippy::cast_precision_loss)]
        let now = current as f32;
        let clipping = luma.clipped > s.clip_share;
        if !clipping {
            self.relax(now_ns, luma);
        }
        let (wanted, increase) = if clipping {
            let over = (luma.clipped - s.clip_share) / s.clip_share.max(f32::EPSILON);
            self.ceiling = Some((current, now_ns));
            (now - now.abs() * (s.max_step * over).clamp(s.fine_step, s.max_step), false)
        } else if (luma.mean - s.target).abs() <= s.deadband {
            self.last = "in the deadband";
            return None;
        } else {
            let ratio = (s.target / luma.mean.max(1.0)).clamp(1.0 - s.max_step, 1.0 + s.max_step);
            (now + now.abs() * (ratio - 1.0), luma.mean < s.target)
        };
        #[allow(clippy::cast_possible_truncation)]
        let mut next = self.range.clamp((wanted.round() as i64).min(self.cap));
        if (increase && next <= current) || (!increase && next >= current) {
            let step = self.range.step.max(1);
            let adjacent = if increase { current.saturating_add(step) } else { current.saturating_sub(step) };
            next = self.range.clamp(adjacent.min(self.cap));
        }
        if let Some(under) = self.under_ceiling().filter(|&u| increase && next > u) {
            if under <= current {
                self.last = "under the clip ceiling";
                return None;
            }
            next = under;
        }
        if next == current {
            // No legal step remains in this direction within the range and frame-period cap.
            self.last = if current >= self.cap { "at the cap" } else { "at a bound" };
            return None;
        }
        self.last = "changed";
        self.last_change_ns = Some(now_ns);
        Some(next)
    }

    /// `value`'s distance to its neighbor `fine_step` away, at least one legal device step.
    fn fine(&self, value: i64) -> i64 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let margin = ((value as f32).abs() * self.settings.fine_step).round() as i64;
        margin.max(self.range.step.max(1))
    }

    /// The highest value an increase may reach: `fine_step` under the ceiling. None without a ceiling.
    fn under_ceiling(&self) -> Option<i64> {
        let (value, _) = self.ceiling?;
        Some(self.range.clamp(value.saturating_sub(self.fine(value))))
    }

    /// Raises the ceiling by `fine_step` once it has held for `relax_ns`, or each interval while the frame clips
    /// under a quarter of `clip_share`. A ceiling raised past the cap is dropped: it no longer limits anything.
    fn relax(&mut self, now_ns: u64, luma: Luma) {
        let Some((value, at)) = self.ceiling else {
            return;
        };
        let (s, age) = (self.settings, now_ns.saturating_sub(at));
        if age >= s.relax_ns || (luma.clipped < s.clip_share / 4.0 && age >= s.interval_ns) {
            let raised = value.saturating_add(self.fine(value));
            self.ceiling = (raised.saturating_sub(self.fine(raised)) < self.cap).then_some((raised, now_ns));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FACECAM: Range = Range { min: 1, max: 2500, step: 1 };
    const FRAME_30: u64 = 33_333_333;
    const UNIT: u64 = 100_000; // V4L2's exposure_time_absolute: 100 µs
    const S: u64 = 1_000_000_000;

    fn lit(mean: f32) -> Luma {
        Luma { mean, clipped: 0.0 }
    }

    /// A room whose frames brighten in proportion to exposure: `k` luma per unit, clipped at 255.
    fn room(k: f32, exposure: i64) -> Luma {
        #[allow(clippy::cast_precision_loss)]
        lit((k * exposure as f32).min(255.0))
    }

    #[test]
    fn a_dark_room_brightens_toward_the_target_without_overshooting() {
        let mut a = AutoExposure::new(FACECAM, UNIT, FRAME_30, Settings::default());
        let (mut e, k) = (60i64, 1.0f32); // luma 60, target 110
        let mut t = 0;
        for _ in 0..20 {
            t += 2 * S;
            if let Some(next) = a.step(t, e, room(k, e)) {
                assert!(next > e && (next - e) as f32 <= 0.2 * e as f32 + 1.0, "{e} -> {next}: one bounded step up");
                e = next;
            }
        }
        let l = room(k, e).mean;
        assert!((l - 110.0).abs() <= 12.0, "settled in the deadband: luma {l} at {e}");
    }

    #[test]
    fn nothing_moves_inside_the_deadband_or_between_intervals() {
        let mut a = AutoExposure::new(FACECAM, UNIT, FRAME_30, Settings::default());
        assert_eq!(a.step(0, 100, lit(115.0)), None, "close enough");
        assert_eq!(a.step(0, 100, lit(40.0)), Some(120));
        assert_eq!(a.step(S, 120, lit(40.0)), None, "only a second since the last change");
        assert_eq!(a.step(2 * S, 120, lit(40.0)), Some(144));
    }

    #[test]
    fn exposure_never_passes_the_frame_period_so_the_frame_rate_holds() {
        let mut a = AutoExposure::new(FACECAM, UNIT, FRAME_30, Settings::default());
        assert_eq!(a.cap(), 333, "33.3 ms at 30 fps, in 100 µs units");
        assert_eq!(a.step(0, 320, lit(20.0)), Some(333), "up to the cap, not past it");
        assert_eq!(a.step(10 * S, 333, lit(20.0)), None, "at the cap: the camera's gain does the rest");
    }

    #[test]
    fn brightness_steers_with_no_frame_rate_cap() {
        // The Facecam's brightness, 0 to 255 at 128: a bright frame steps it down, a dark one up past any frame
        // period (it has none).
        let brightness = Range { min: 0, max: 255, step: 1 };
        let mut a = AutoExposure::uncapped(brightness, Settings::default());
        assert_eq!(a.step(0, 128, lit(152.0)), Some(102));
        assert_eq!(a.step(10 * S, 220, lit(40.0)), Some(255), "up to the control's top, not a frame period");
    }

    #[test]
    fn low_brightness_values_can_leave_zero_and_cross_rounding_traps() {
        for current in [0, 1, 2] {
            let mut a = AutoExposure::uncapped(Range { min: 0, max: 255, step: 1 }, Settings::default());
            assert_eq!(a.step(0, current, lit(45.0)), Some(current + 1), "dark at {current}");
        }
        let mut a = AutoExposure::uncapped(Range { min: 0, max: 255, step: 1 }, Settings::default());
        assert_eq!(a.step(0, 1, lit(200.0)), Some(0), "one legal step down");
        assert_eq!(a.step(10 * S, 0, lit(200.0)), None, "at the real minimum");
    }

    #[test]
    fn coarse_and_signed_controls_move_in_the_requested_direction() {
        let coarse = Range { min: 0, max: 25, step: 10 };
        let mut a = AutoExposure::uncapped(coarse, Settings::default());
        assert_eq!(a.step(0, 0, lit(45.0)), Some(10));
        assert_eq!(a.step(10 * S, 10, lit(45.0)), Some(20));
        assert_eq!(a.step(20 * S, 20, lit(45.0)), None, "25 is not a legal control step");
        assert_eq!(a.step(30 * S, 20, lit(200.0)), Some(10));
        let mut b = AutoExposure::uncapped(Range { min: -64, max: 64, step: 1 }, Settings::default());
        assert_eq!(b.step(0, -10, lit(45.0)), Some(-8), "brightening increases a signed control");
        assert_eq!(b.step(10 * S, -10, lit(200.0)), Some(-12), "dimming decreases it");
        assert_eq!(b.step(20 * S, 0, Luma { mean: 90.0, clipped: 0.2 }), Some(-1), "clipping still dims");
    }

    #[test]
    fn a_quantized_step_cannot_cross_the_frame_cap() {
        let mut a = AutoExposure::new(Range { min: 0, max: 100, step: 10 }, UNIT, 25 * UNIT, Settings::default());
        assert_eq!(a.cap(), 20, "the frame cap is a legal device value");
        assert_eq!(a.step(0, 10, lit(45.0)), Some(20));
        assert_eq!(a.step(10 * S, 20, lit(45.0)), None);
        assert_eq!(a.last(), "at the cap");
    }

    #[test]
    fn a_change_the_camera_refused_is_said_so_not_changed() {
        let mut a = AutoExposure::new(FACECAM, UNIT, FRAME_30, Settings::default());
        assert!(a.step(0, 156, lit(150.0)).is_some());
        assert_eq!(a.last(), "changed");
        a.refused();
        assert_eq!(a.last(), "refused by the camera");
    }

    #[test]
    fn a_clipped_frame_steps_down_whatever_its_mean() {
        let mut a = AutoExposure::new(FACECAM, UNIT, FRAME_30, Settings::default());
        assert_eq!(a.step(0, 200, Luma { mean: 90.0, clipped: 0.2 }), Some(160));
    }

    /// cam0 in #3149, from its log: brightness 125 gives luma 82 and 174 gives 113, and a lamp in the 360 view
    /// clips 2.7% of the frame at 128, passing 5% near 152. No brightness has the mean in the deadband (98 at 150)
    /// with under 5% clipped. `jitter` flutters the clipped share by a frame-to-frame 0.3%.
    fn cam0(brightness: i64, jitter: bool) -> Luma {
        #[allow(clippy::cast_precision_loss)]
        let b = brightness as f32;
        let flutter = if jitter { 0.003 } else { 0.0 };
        Luma { mean: (0.63 * b + 3.0).min(255.0), clipped: (0.027 + (b - 128.0) * 0.000_96 + flutter).clamp(0.0, 1.0) }
    }

    /// Steps `a` once a second for `seconds` from `start` against `camera`, returning the values it set, by second.
    fn run(a: &mut AutoExposure, start: i64, seconds: u64, camera: impl Fn(i64, u64) -> Luma) -> Vec<(u64, i64)> {
        let (mut b, mut changes) = (start, Vec::new());
        for t in 0..seconds {
            if let Some(next) = a.step(t * S, b, camera(b, t)) {
                b = next;
                changes.push((t, b));
            }
        }
        changes
    }

    #[test]
    fn a_lamp_that_clips_below_the_target_settles_under_it_instead_of_hunting() {
        let brightness = Range { min: 0, max: 255, step: 1 };
        let mut a = AutoExposure::uncapped(brightness, Settings::default());
        // Starting where the log did, an hour of frames, the clipped share fluttering every other one.
        let changes = run(&mut a, 155, 3600, |b, t| cam0(b, t % 2 == 0));
        let settled: Vec<_> = changes.iter().filter(|&&(t, _)| t > 10).collect();
        assert!(changes.len() <= 10, "a few changes an hour, not 1800: {changes:?}");
        assert!(changes.iter().take_while(|&&(t, _)| t <= 10).count() <= 3, "settled in a few steps: {changes:?}");
        for &&(_, b) in &settled {
            assert!((140..=160).contains(&b), "stays near the clip point, not 20% swings: {changes:?}");
        }
        let (_, last) = *changes.last().expect("it moved off the clipping start");
        assert!(cam0(last, true).clipped <= 0.05, "held where it doesn't clip: {last}");
        assert!(matches!(a.last(), "under the clip ceiling" | "waiting"), "{}", a.last());
    }

    #[test]
    fn the_ceiling_lifts_once_the_lamp_goes_off() {
        let brightness = Range { min: 0, max: 255, step: 1 };
        let mut a = AutoExposure::uncapped(brightness, Settings::default());
        let held = run(&mut a, 155, 60, |b, _| cam0(b, false)).last().expect("it stepped down").1;
        // The lamp goes off: nothing clips, and the room is a little darker.
        #[allow(clippy::cast_precision_loss)]
        let dark = |b: i64, _| Luma { mean: (0.6 * b as f32).min(255.0), clipped: 0.0 };
        let changes = run(&mut a, held, 120, dark);
        let (_, last) = *changes.last().expect("it brightened");
        assert!((dark(last, 0).mean - 110.0).abs() <= 12.0, "back in the deadband: {held} then {changes:?}");
    }

    #[test]
    fn a_frame_just_over_the_clip_share_steps_down_finely() {
        let mut a = AutoExposure::new(FACECAM, UNIT, FRAME_30, Settings::default());
        assert_eq!(a.step(0, 200, Luma { mean: 105.0, clipped: 0.053 }), Some(190), "5%, not 20%");
        assert_eq!(a.step(2 * S, 190, Luma { mean: 90.0, clipped: 0.04 }), None, "not back up to 200");
        assert_eq!(a.last(), "under the clip ceiling");
        assert_eq!(a.step(4 * S, 190, Luma { mean: 105.0, clipped: 0.075 }), Some(171), "half again over: 10%");
    }

    #[test]
    fn values_stay_on_the_controls_steps_and_within_its_range() {
        let coarse = Range { min: 10, max: 300, step: 10 };
        let mut a = AutoExposure::new(coarse, UNIT, FRAME_30, Settings::default());
        assert_eq!(a.step(0, 100, lit(40.0)), Some(120));
        let mut b = AutoExposure::new(coarse, UNIT, FRAME_30, Settings::default());
        assert_eq!(b.step(0, 10, lit(250.0)), None, "at the minimum already");
    }

    #[test]
    fn luma_reads_a_plane_or_every_other_byte_of_a_packed_frame() {
        assert_eq!(luma(&[100, 200, 250, 50], 1), Some(Luma { mean: 150.0, clipped: 0.25 }));
        // UYVY: U Y V Y, so the Y samples are every second byte from the second.
        assert_eq!(luma(&[128, 10, 128, 30][1..], 2), Some(Luma { mean: 20.0, clipped: 0.0 }));
        assert_eq!(luma(&[], 1), None);
    }
}
