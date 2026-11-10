//! The choices the AVFoundation source makes, as pure functions so they're tested on any OS: a camera's USB ids from
//! its model id, FourCC codes, and which of its formats to use.
//!
//! AVFoundation's video data output converts what the camera sends (its raw formats, or MJPEG, listed as `dmb1`) to
//! the pixel format it's asked for, and it can't hand over a camera's MJPEG as is, so on macOS a camera is captured as
//! packed 4:2:2 (`2vuy`, which is UYVY) whatever the manifest asks.

use crate::manifest::VideoSettings;
use crate::video::{Mode, ladder};

/// The FourCC macOS cameras are captured in: `2vuy` (`kCVPixelFormatType_422YpCbCr8`), UYVY byte order.
pub const CAPTURE_FOURCC: [u8; 4] = *b"2vuy";

/// The shape name a stream captured as [`CAPTURE_FOURCC`] registers.
pub const CAPTURE_SHAPE: &str = "UYVY";

/// A four-character code (`OSType`, `FourCharCode`) as its four bytes, most significant first.
#[must_use]
pub fn fourcc_bytes(code: u32) -> [u8; 4] {
    code.to_be_bytes()
}

/// The USB vendor and product in a UVC camera's AVFoundation model id, like `UVC Camera VendorID_4057
/// ProductID_120` (decimal). `None` for a camera that isn't UVC (the built-in one, Continuity Camera).
#[must_use]
pub fn usb_from_model_id(model: &str) -> Option<(u16, u16)> {
    let number = |key: &str| {
        let at = model.find(key)? + key.len();
        let digits: String = model[at..].chars().take_while(char::is_ascii_digit).collect();
        digits.parse::<u16>().ok()
    };
    Some((number("VendorID_")?, number("ProductID_")?))
}

/// One of a camera's formats: its mode and the highest frame rate it supports.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Offered {
    /// Its FourCC and size.
    pub mode: Mode,
    /// Its highest frame rate.
    pub max_fps: f64,
}

/// The camera format to open: the first size on the manifest's ladder the camera offers in any format (the output
/// converts it to [`CAPTURE_FOURCC`]), preferring native USB-compressed MJPEG (`dmb1`) among those that reach the
/// manifest's frame rate (within half a frame a second), then one already in [`CAPTURE_FOURCC`] (no conversion), then the first offered. `None` when the camera
/// offers none of the ladder's sizes.
#[must_use]
pub fn pick_format(offered: &[Offered], video: &VideoSettings) -> Option<Offered> {
    let want_fps = f64::from(video.fps) - 0.5;
    for (width, height) in ladder(video) {
        let best = offered
            .iter()
            .filter(|o| o.mode.width == width && o.mode.height == height && o.max_fps.is_finite() && o.max_fps > 0.0)
            .min_by_key(|o| (o.max_fps < want_fps, o.mode.fourcc != *b"dmb1", o.mode.fourcc != CAPTURE_FOURCC));
        if let Some(best) = best {
            return Some(*best);
        }
    }
    None
}

/// Verifies the delivered native output before any frame reaches capture's publisher.
#[cfg(any(target_os = "macos", test))]
pub(crate) fn validate_frame(expected: Mode, delivered: Mode, stride: u32, bytes: usize) -> Result<(), String> {
    if delivered != expected {
        return Err(format!(
            "requested {}x{} {:?}, camera delivered {}x{} {:?}; refusing unexpected capture format",
            expected.width, expected.height, expected.fourcc, delivered.width, delivered.height, delivered.fourcc
        ));
    }
    let row = expected.width.checked_mul(2).ok_or("capture width overflow")?;
    let maximum_row = row.checked_add(127).map(|row| row / 128 * 128).ok_or("admitted capture stride overflow")?;
    if expected.width == 0 || expected.height == 0 || stride < row || stride > maximum_row {
        return Err("native UYVY frame dimensions/stride exceed the admitted 128-byte row layout".into());
    }
    let required = usize::try_from(stride)
        .ok()
        .and_then(|row| usize::try_from(expected.height).ok().and_then(|height| row.checked_mul(height)))
        .ok_or("capture frame length overflow")?;
    if bytes != required {
        return Err(format!("native UYVY frame has {bytes} bytes, expected {required}"));
    }
    Ok(())
}

/// Delivered geometry is latched once, while the authorized native-format budget remains unchanged.
#[cfg(any(target_os = "macos", test))]
pub(crate) struct DeliveredLayout {
    requested: Mode,
    duration: (i64, i32),
    share: u64,
    native_rate: u64,
    accepted: Option<(Mode, u32)>,
}

#[cfg(any(target_os = "macos", test))]
impl DeliveredLayout {
    pub(crate) fn new(requested: Mode, duration: (i64, i32), share: u64, native_rate: u64) -> Self {
        Self { requested, duration, share, native_rate, accepted: None }
    }

    pub(crate) fn observe(&mut self, delivered: Mode, stride: u32, bytes: usize) -> Result<Mode, String> {
        if !valid_duration(self.duration) || self.share == 0 || self.native_rate == 0 || self.native_rate > self.share {
            return Err("verified native format/rate exceeds its immutable authorized share".into());
        }
        let expected = match self.accepted {
            Some((accepted, _)) => accepted,
            None => {
                if delivered.fourcc != self.requested.fourcc
                    || delivered.width != self.requested.width
                    || delivered.height == 0
                    || delivered.height > self.requested.height
                {
                    return Err("first frame may only reduce height at the requested width and pixel format".into());
                }
                delivered
            }
        };
        validate_frame(expected, delivered, stride, bytes)?;
        if u32::try_from(bytes).is_err() {
            return Err("native frame length cannot be represented in capture metadata".into());
        }
        if self.accepted.is_some_and(|(_, accepted)| accepted != stride) {
            return Err("native frame stride changed during capture".into());
        }
        let value = u128::try_from(self.duration.0).map_err(|_| "invalid verified frame duration")?;
        let scale = u128::try_from(self.duration.1).map_err(|_| "invalid verified frame rate")?;
        let numerator = (bytes as u128)
            .checked_mul(scale)
            .and_then(|rate| rate.checked_add(value - 1))
            .ok_or("actual frame-rate arithmetic overflow")?;
        let actual_rate = numerator / value;
        if actual_rate > u128::from(self.share) {
            return Err("actual delivered stride/height/rate exceeds its authorized share".into());
        }
        self.accepted = Some((delivered, stride));
        Ok(delivered)
    }
}

/// Ordered native duration writes; the backend executes this same pure sequence under its device lock.
#[cfg(any(target_os = "macos", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DurationWrite {
    Minimum((i64, i32)),
    Maximum((i64, i32)),
}

#[cfg(any(target_os = "macos", test))]
pub(crate) fn duration_writes(ranges: &[RateRange], requested: (i64, i32), hold: bool) -> Option<Vec<DurationWrite>> {
    if !valid_duration(requested) {
        return None;
    }
    let supported: Vec<_> = ranges
        .iter()
        .filter(|range| {
            range.min_fps.is_finite()
                && range.max_fps.is_finite()
                && range.min_fps > 0.0
                && range.max_fps >= range.min_fps
                && valid_duration(range.min_duration)
                && valid_duration(range.max_duration)
                && duration_le(range.min_duration, range.max_duration)
        })
        .collect();
    if !supported
        .iter()
        .any(|range| duration_le(range.min_duration, requested) && duration_le(requested, range.max_duration))
    {
        return None;
    }
    let longest = supported.iter().map(|range| range.max_duration).max_by(|left, right| {
        (i128::from(left.0) * i128::from(right.1)).cmp(&(i128::from(right.0) * i128::from(left.1)))
    })?;
    let mut writes = vec![DurationWrite::Maximum(longest), DurationWrite::Minimum(requested)];
    if hold {
        writes.push(DurationWrite::Maximum(requested));
    }
    Some(writes)
}

/// One of a format's frame rate ranges, with its durations as `CMTime` (value, timescale).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RateRange {
    /// The slowest rate, frames a second.
    pub min_fps: f64,
    /// The fastest rate, frames a second.
    pub max_fps: f64,
    /// The shortest frame duration (at `max_fps`).
    pub min_duration: (i64, i32),
    /// The longest frame duration (at `min_fps`).
    pub max_duration: (i64, i32),
}

/// The frame duration to set for `fps`: one the format supports, since AVFoundation throws on any other. Exactly
/// `1/fps` when a range includes `fps`; else the fastest range below it, at its own shortest duration (a format
/// at 29.97 for an ask of 30 gets 1001/30000, not 1/30); else, when every range is faster, the slowest range's
/// longest duration is refused: capture never silently exceeds the requested rate. `None` for unusable ranges.
#[must_use]
pub fn frame_duration(ranges: &[RateRange], fps: u32) -> Option<(i64, i32)> {
    let scale = i32::try_from(fps).ok().filter(|scale| *scale > 0)?;
    let want = f64::from(fps);
    let valid = |range: &&RateRange| {
        range.min_fps.is_finite()
            && range.max_fps.is_finite()
            && range.min_fps > 0.0
            && range.max_fps >= range.min_fps
            && valid_duration(range.min_duration)
            && valid_duration(range.max_duration)
            && duration_le(range.min_duration, range.max_duration)
    };
    let supported: Vec<_> = ranges.iter().filter(valid).collect();
    let requested = (1, scale);
    if supported.iter().any(|range| {
        range.min_fps <= want
            && want <= range.max_fps
            && duration_le(range.min_duration, requested)
            && duration_le(requested, range.max_duration)
    }) {
        return Some(requested);
    }
    supported
        .into_iter()
        .filter(|range| range.max_fps <= want && duration_le(requested, range.min_duration))
        .max_by(|a, b| a.max_fps.total_cmp(&b.max_fps))
        .map(|range| range.min_duration)
}

fn valid_duration(time: (i64, i32)) -> bool {
    time.0 > 0 && time.1 > 0
}

fn duration_le(left: (i64, i32), right: (i64, i32)) -> bool {
    i128::from(left.0) * i128::from(right.1) <= i128::from(right.0) * i128::from(left.1)
}

/// One native device format and its actual supported frame-rate intervals, for bounded admission planning.
#[derive(Clone, Debug, PartialEq)]
pub struct CameraFormat {
    /// Native FourCC, dimensions, and maximum rate.
    pub offered: Offered,
    /// Supported rates and their exact device durations.
    pub rates: Vec<RateRange>,
}

/// Whether a device readback retained the selected frame duration, including its validity flags.
#[cfg(any(target_os = "macos", test))]
pub(crate) fn duration_matches(actual: (i64, i32), expected: (i64, i32)) -> bool {
    valid_duration(actual) && valid_duration(expected) && duration_le(actual, expected) && duration_le(expected, actual)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn o(f: &[u8; 4], width: u32, height: u32, max_fps: f64) -> Offered {
        Offered { mode: Mode { fourcc: *f, width, height }, max_fps }
    }

    #[test]
    fn a_uvc_model_id_gives_the_usb_ids() {
        assert_eq!(usb_from_model_id("UVC Camera VendorID_4057 ProductID_120"), Some((0x0fd9, 0x0078)));
        assert_eq!(usb_from_model_id("FaceTime HD Camera"), None);
        assert_eq!(usb_from_model_id("UVC Camera VendorID_99999 ProductID_1"), None, "not a u16");
    }

    #[test]
    fn fourcc_codes_read_most_significant_first() {
        assert_eq!(fourcc_bytes(0x3276_7579), *b"2vuy");
        assert_eq!(fourcc_bytes(u32::from_be_bytes(*b"420v")), *b"420v");
    }

    #[test]
    fn the_format_follows_the_ladder_in_2vuy_and_prefers_one_at_the_frame_rate() {
        let video = VideoSettings::default(); // 1280x720 at 30, then 960x540, then 640x480
        let offered = [
            o(b"420v", 1280, 720, 30.0),
            o(b"2vuy", 1280, 720, 10.0),
            o(b"2vuy", 1280, 720, 30.0),
            o(b"2vuy", 640, 480, 30.0),
        ];
        assert_eq!(pick_format(&offered, &video), Some(o(b"2vuy", 1280, 720, 30.0)));
        // Only a slow format at the first size: still the first size (the frame rate is the camera's to give).
        let slow = [o(b"2vuy", 1280, 720, 10.0), o(b"2vuy", 640, 480, 30.0)];
        assert_eq!(pick_format(&slow, &video), Some(o(b"2vuy", 1280, 720, 10.0)));
        // Not at the first size: the ladder's next.
        assert_eq!(pick_format(&[o(b"2vuy", 960, 540, 29.97)], &video).map(|f| f.mode.width), Some(960));
        // Any native format at the size will do (the output converts); 2vuy is preferred only among equals.
        assert_eq!(pick_format(&[o(b"yuvs", 1280, 720, 30.0)], &video), Some(o(b"yuvs", 1280, 720, 30.0)));
        assert_eq!(
            pick_format(&[o(b"dmb1", 1280, 720, 30.0), o(b"2vuy", 1280, 720, 30.0)], &video),
            Some(o(b"dmb1", 1280, 720, 30.0))
        );
        assert_eq!(pick_format(&[o(b"2vuy", 1920, 1080, 30.0)], &video), None, "no size on the ladder");
    }

    fn range(min_fps: f64, max_fps: f64, min_d: (i64, i32), max_d: (i64, i32)) -> RateRange {
        RateRange { min_fps, max_fps, min_duration: min_d, max_duration: max_d }
    }

    #[test]
    fn the_frame_duration_is_always_one_the_format_supports() {
        let continuous = [range(5.0, 30.0, (1, 30), (1, 5))];
        assert_eq!(frame_duration(&continuous, 30), Some((1, 30)));
        assert_eq!(frame_duration(&continuous, 25), Some((1, 25)));
        // Discrete rates at 30 and 15: an ask of 25 gets the fastest below it, 15.
        let discrete = [range(30.0, 30.0, (1, 30), (1, 30)), range(15.0, 15.0, (1, 15), (1, 15))];
        assert_eq!(frame_duration(&discrete, 25), Some((1, 15)));
        // 29.97 for an ask of 30: its own duration, not 1/30.
        let ntsc = [range(29.97, 29.97, (1001, 30000), (1001, 30000))];
        assert_eq!(frame_duration(&ntsc, 30), Some((1001, 30000)));
        // Every range faster than the ask: the slowest one's longest duration.
        assert_eq!(frame_duration(&[range(30.0, 60.0, (1, 60), (1, 30))], 10), None);
        assert_eq!(frame_duration(&[], 30), None);
    }
    #[test]
    fn new_rules_prefer_native_usb_compression_at_the_requested_size_and_rate() {
        let offered = [o(b"2vuy", 1280, 720, 30.0), o(b"dmb1", 1280, 720, 30.0)];
        assert_eq!(pick_format(&offered, &VideoSettings::default()).unwrap().mode.fourcc, *b"dmb1");
    }

    #[test]
    fn new_rules_reject_oversized_or_wrong_pixel_format_frames() {
        let expected = Mode { fourcc: CAPTURE_FOURCC, width: 1280, height: 720 };
        assert!(validate_frame(expected, expected, 2560, 2560 * 720).is_ok());
        assert!(validate_frame(expected, Mode { width: 1920, height: 1080, ..expected }, 3840, 3840 * 1080).is_err());
        assert!(validate_frame(expected, Mode { fourcc: *b"420v", ..expected }, 2560, 2560 * 720).is_err());
        assert!(validate_frame(expected, expected, 1280, 1280 * 720).is_err());
        assert!(validate_frame(expected, expected, 2560, 2560 * 720 - 1).is_err());
    }

    #[test]
    fn new_rules_never_raise_rate_above_the_request_or_accept_invalid_durations() {
        assert_eq!(frame_duration(&[range(30.0, 60.0, (1, 60), (1, 30))], 10), None);
        assert_eq!(frame_duration(&[range(0.0, 30.0, (0, 30), (1, 1))], 30), None);
        assert_eq!(frame_duration(&[range(5.0, 30.0, (1, 30), (1, 5))], 0), None);
    }

    #[test]
    fn duration_readback_compares_exact_rational_values_and_rejects_invalid_times() {
        assert!(duration_matches((2, 60), (1, 30)));
        assert!(!duration_matches((1, 60), (1, 30)));
        assert!(!duration_matches((0, 30), (1, 30)));
        assert!(!duration_matches((1, 0), (1, 30)));
    }
    #[test]
    fn new_rules_native_stride_cannot_exceed_the_admitted_128_byte_row_bound() {
        let expected = Mode { fourcc: CAPTURE_FOURCC, width: 1280, height: 720 };
        assert!(
            validate_frame(expected, expected, 2688, 2688 * 720).is_err(),
            "larger native rows must not bypass the admitted bandwidth estimate"
        );
    }
    #[test]
    fn first_layout_policy_accepts_smaller_height_then_freezes_all_geometry() {
        let requested = Mode { fourcc: CAPTURE_FOURCC, width: 640, height: 480 };
        let delivered = Mode { height: 360, ..requested };
        let mut layout = DeliveredLayout::new(requested, (1, 30), 20_000_000, 18_432_000);
        assert_eq!(layout.observe(delivered, 1280, 1280 * 360), Ok(delivered));
        assert!(layout.observe(delivered, 1280, 1280 * 360).is_ok());
        assert!(layout.observe(requested, 1280, 1280 * 480).is_err(), "later taller frames must be refused");
        assert!(
            layout.observe(Mode { height: 240, ..requested }, 1280, 1280 * 240).is_err(),
            "adaptation is first-frame only"
        );
    }

    #[test]
    fn duration_policy_widens_max_before_slowing_without_hold_or_speeding_with_hold() {
        let ranges = [range(5.0, 60.0, (1, 60), (1, 5))];
        assert_eq!(
            duration_writes(&ranges, (1, 10), false),
            Some(vec![DurationWrite::Maximum((1, 5)), DurationWrite::Minimum((1, 10))])
        );
        assert_eq!(
            duration_writes(&ranges, (1, 60), true),
            Some(vec![
                DurationWrite::Maximum((1, 5)),
                DurationWrite::Minimum((1, 60)),
                DurationWrite::Maximum((1, 60))
            ])
        );
    }
    #[test]
    fn first_layout_revalidates_share_without_discounting_native_cost_and_rejects_invalid_metadata() {
        let requested = Mode { fourcc: CAPTURE_FOURCC, width: 640, height: 480 };
        let delivered = Mode { height: 360, ..requested };
        let make = |share, native| DeliveredLayout::new(requested, (1, 30), share, native);
        assert!(
            make(20_000_000, 24_000_000).observe(delivered, 1280, 1280 * 360).is_err(),
            "cropped output must not discount the configured native USB format"
        );
        assert!(
            make(13_000_000, 12_000_000).observe(delivered, 1280, 1280 * 360).is_err(),
            "actual output still needs to fit its share"
        );
        for mode in [
            Mode { width: 800, ..delivered },
            Mode { height: 481, ..delivered },
            Mode { height: 0, ..delivered },
            Mode { fourcc: *b"420v", ..delivered },
        ] {
            assert!(make(20_000_000, 18_432_000).observe(mode, 1280, 1280 * 360).is_err());
        }
        assert!(make(20_000_000, 18_432_000).observe(delivered, 1408, 1408 * 360).is_err());
        assert!(make(20_000_000, 18_432_000).observe(delivered, 1280, 1280 * 360 - 1).is_err());
        assert!(
            DeliveredLayout::new(requested, (0, 30), 20_000_000, 18_432_000)
                .observe(delivered, 1280, 1280 * 360)
                .is_err()
        );
        let odd_row = Mode { width: 642, ..requested };
        let actual = Mode { height: 360, ..odd_row };
        let mut layout = DeliveredLayout::new(odd_row, (1, 30), 30_000_000, 21_000_000);
        assert!(layout.observe(actual, 1344, 1344 * 360).is_ok());
        assert!(
            layout.observe(actual, 1408, 1408 * 360).is_err(),
            "later stride changes are forbidden even inside the admission bound"
        );
    }

    #[test]
    fn duration_limits_reject_invalid_or_unsupported_ranges() {
        assert!(duration_writes(&[], (1, 10), false).is_none());
        assert!(duration_writes(&[range(30.0, 30.0, (1, 30), (1, 30))], (1, 10), false).is_none());
        assert!(duration_writes(&[range(5.0, 30.0, (0, 30), (1, 5))], (1, 10), false).is_none());
    }
}
