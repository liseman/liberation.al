//! Equal shares of one conservative unknown-bus budget, without device I/O.
//!
//! All configured cameras, including absent ones, retain fixed shares. Compressed native formats are preferred,
//! but unknown compression ratios earn no bandwidth discount. Known packed, subsampled, and compressed formats cost
//! at least two bytes/pixel; wider RGB and unknown native formats cost four, with 128-byte padded rows.
//! Linux does not use this policy. A verified faster transport needs an explicit budget override.

use crate::avf_rules::{CAPTURE_FOURCC, CameraFormat, frame_duration};
use crate::manifest::{DeviceKind, Manifest, VideoSettings};
use crate::video::ladder;

/// Decimal bytes/second below USB 2's theoretical signaling ceiling, leaving protocol/audio room.
/// A conservative policy estimate, not a measurement or a usable-bandwidth guarantee.
pub const DEFAULT_BUDGET_BYTES_PER_SECOND: u64 = 40_000_000;

/// Native choice and effective worker settings; the persisted manifest remains unchanged.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedCamera {
    /// Exact manifest camera identity supplied by the caller.
    pub name: String,
    /// Exact selected size/rate cap; fallback is cleared to prevent opening an unbudgeted mode.
    pub video: VideoSettings,
    /// The actually offered native format and its rate ranges.
    pub native: CameraFormat,
    /// Validated native frame duration as (value, timescale), never faster than the selected cap.
    pub frame_duration: (i64, i32),
    /// Native-format proxy at the whole-number cap: 29.97 fps is charged at 30, wide/unknown pixels at four bytes.
    pub bytes_per_second: u64,
    /// Immutable equal share authorized by the parent supervisor.
    pub share_bytes_per_second: u64,
}

/// A supervisor fixes its share for its lifetime; budget or camera-count changes need a supervisor restart.
/// # Errors
/// Zero budget, or insufficient budget to give every configured camera even one byte/second.
pub fn manifest_share(manifest: &Manifest) -> Result<u64, String> {
    let budget = manifest.supervisor.mac_raw_budget_bytes_per_second;
    if budget == 0 {
        return Err("[supervisor].mac_raw_budget_bytes_per_second must be positive".into());
    }
    let cameras = manifest.devices.iter().filter(|d| d.enabled && d.kind == DeviceKind::Camera).count();
    let count = u64::try_from(cameras.max(1)).map_err(|_| "too many configured cameras")?;
    let share = budget / count;
    if share == 0 {
        return Err("macOS raw bandwidth budget is too small for the configured camera count".into());
    }
    Ok(share)
}

fn raw_rate(width: u32, height: u32, fps: u32, bytes_per_pixel: u64) -> Result<u64, String> {
    if width == 0 || height == 0 || fps == 0 {
        return Err("macOS raw bandwidth needs positive width, height, and fps".into());
    }
    let row = u64::from(width).checked_mul(bytes_per_pixel).and_then(|n| n.checked_add(127)).map(|n| n / 128 * 128);
    row.and_then(|n| n.checked_mul(u64::from(height)))
        .and_then(|n| n.checked_mul(u64::from(fps)))
        .ok_or_else(|| "macOS raw camera bandwidth arithmetic overflow".into())
}

fn compressed(fourcc: [u8; 4]) -> bool {
    matches!(&fourcc, b"dmb1" | b"MJPG" | b"jpeg")
}

fn native_pixel_bytes(fourcc: [u8; 4]) -> u64 {
    match &fourcc {
        b"2vuy" | b"UYVY" | b"yuvs" | b"YUY2" | b"420v" | b"420f" | b"NV12" | b"I420" | b"dmb1" | b"MJPG" | b"jpeg" => {
            2
        }
        _ => 4, // Includes BGRA/ARGB/RGBA/ABGR, 24-bit RGB (rounded up), and unknown native transports.
    }
}

fn duration_at_cap(format: &CameraFormat, fps: u32) -> Option<(i64, i32)> {
    let duration = frame_duration(&format.rates, fps)?;
    if duration.0 <= 0 || duration.1 <= 0 {
        return None;
    }
    // A fixed-30 format must not masquerade as 15 fps, even if a duration helper clamps upward.
    if u128::from(u32::try_from(duration.1).ok()?) > u128::try_from(duration.0).ok()? * u128::from(fps) {
        return None;
    }
    let actual = f64::from(duration.1) / duration.0 as f64;
    (actual >= f64::from(fps) - 0.5).then_some(duration)
}

/// Try every configured ladder size at requested fps, then at 15 and 10 (never above requested fps).
/// Neither dimension may exceed the original request, even if a larger default fallback is configured.
/// Only actual inventory size/rate ranges within this camera's fixed share are eligible. No devices are opened.
/// # Errors
/// Invalid/overflowing requirements, or no offered size/rate fitting the share.
pub fn plan_camera(
    name: &str,
    video: &VideoSettings,
    inventory: &[CameraFormat],
    share_bytes_per_second: u64,
) -> Result<PlannedCamera, String> {
    if video.fps == 0 || share_bytes_per_second == 0 {
        return Err(format!("{name}: camera fps and macOS raw bandwidth share must be positive"));
    }
    let mut caps = Vec::new();
    for fps in [video.fps, video.fps.min(15), video.fps.min(10)] {
        if !caps.contains(&fps) {
            caps.push(fps);
        }
    }
    for fps in caps {
        for (width, height) in ladder(video) {
            if width > video.width || height > video.height {
                continue;
            }
            if !inventory.iter().any(|f| f.offered.mode.width == width && f.offered.mode.height == height) {
                continue;
            }
            let choice = inventory
                .iter()
                .filter(|f| f.offered.mode.width == width && f.offered.mode.height == height)
                .filter_map(|f| {
                    let duration = duration_at_cap(f, fps)?;
                    // Cost each actual native format. An overflowing mode cannot fit, but a safe fallback may.
                    let rate = raw_rate(width, height, fps, native_pixel_bytes(f.offered.mode.fourcc)).ok()?;
                    (rate <= share_bytes_per_second).then_some((f, duration, rate))
                })
                .min_by_key(|(f, _, _)| {
                    (
                        !compressed(f.offered.mode.fourcc),
                        native_pixel_bytes(f.offered.mode.fourcc),
                        f.offered.mode.fourcc != CAPTURE_FOURCC,
                    )
                });
            if let Some((native, duration, rate)) = choice {
                let mut selected = video.clone();
                selected.width = width;
                selected.height = height;
                selected.fps = fps;
                selected.fallback.clear();
                return Ok(PlannedCamera {
                    name: name.into(),
                    video: selected,
                    native: native.clone(),
                    frame_duration: duration,
                    bytes_per_second: rate,
                    share_bytes_per_second,
                });
            }
        }
    }
    Err(format!(
        "{name}: no offered size/rate fits equal macOS raw share {share_bytes_per_second} B/s; all configured cameras share an unknown bus. Lower size/fps or explicitly raise [supervisor].mac_raw_budget_bytes_per_second only for a verified faster bus"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avf_rules::{Offered, RateRange};
    use crate::video::Mode;

    fn format(width: u32, height: u32, min: i32, max: i32) -> CameraFormat {
        CameraFormat {
            offered: Offered { mode: Mode { fourcc: CAPTURE_FOURCC, width, height }, max_fps: f64::from(max) },
            rates: vec![RateRange {
                min_fps: f64::from(min),
                max_fps: f64::from(max),
                min_duration: (1, max),
                max_duration: (1, min),
            }],
        }
    }
    fn inventory() -> Vec<CameraFormat> {
        vec![format(1280, 720, 1, 30), format(960, 540, 1, 30), format(640, 480, 1, 30)]
    }
    fn cameras(count: usize, budget: Option<u64>) -> Manifest {
        let mut text = String::new();
        if let Some(budget) = budget {
            text.push_str(&format!("[supervisor]\nmac_raw_budget_bytes_per_second={budget}\n"));
        }
        for index in 0..count {
            text.push_str(&format!("[[device]]\nname='cam{index}'\nkind='camera'\nmatch={{name='Camera'}}\n"));
        }
        Manifest::parse(&text).unwrap()
    }

    #[test]
    fn three_cameras_keep_equal_640_by_480_at_15_shares_without_borrowing() {
        let manifest = cameras(3, None);
        let share = manifest_share(&manifest).unwrap();
        assert_eq!(share, 13_333_333);
        let mut total = 0;
        for camera in &manifest.devices {
            let plan = plan_camera(&camera.name, &camera.video().unwrap(), &inventory(), share).unwrap();
            assert_eq!((plan.video.width, plan.video.height, plan.video.fps), (640, 480, 15));
            assert!(plan.video.fallback.is_empty());
            assert_eq!(plan.bytes_per_second, 9_216_000);
            total += plan.bytes_per_second;
        }
        assert_eq!(total, 27_648_000);
    }
    #[test]
    fn absent_cameras_keep_their_share_and_microphones_do_not_count() {
        let mut manifest = cameras(3, None);
        let mic = Manifest::parse("[[device]]\nname='mic'\nkind='microphone'\nmatch={name='Mic'}").unwrap();
        manifest.devices.extend(mic.devices);
        assert_eq!(manifest_share(&manifest).unwrap(), 13_333_333);
        assert_eq!(
            plan_camera("only-attached", &VideoSettings::default(), &inventory(), manifest_share(&manifest).unwrap())
                .unwrap()
                .video
                .fps,
            15
        );
    }
    #[test]
    fn all_sizes_are_tried_at_requested_fps_before_reducing_rate() {
        let plan = plan_camera("cam", &VideoSettings::default(), &inventory(), 40_000_000).unwrap();
        assert_eq!((plan.video.width, plan.video.height, plan.video.fps), (960, 540, 30));
        assert_eq!(plan.bytes_per_second, 31_104_000);
    }
    #[test]
    fn fixed_30_inventory_cannot_pretend_to_run_at_15_or_10() {
        assert!(plan_camera("cam", &VideoSettings::default(), &[format(640, 480, 30, 30)], 13_333_333).is_err());
    }
    #[test]
    fn native_compression_is_preferred_without_a_bandwidth_discount() {
        let mut native = format(960, 540, 1, 30);
        native.offered.mode.fourcc = *b"dmb1";
        let mut formats = inventory();
        formats.push(native);
        let plan = plan_camera("cam", &VideoSettings::default(), &formats, 40_000_000).unwrap();
        assert_eq!(plan.native.offered.mode.fourcc, *b"dmb1");
        assert_eq!(plan.bytes_per_second, 31_104_000);
    }

    fn wide_native_format_uses_its_larger_cost(fourcc: [u8; 4]) {
        let mut native = format(640, 480, 1, 30);
        native.offered.mode.fourcc = fourcc;
        let plan = plan_camera("cam", &VideoSettings::default(), &[native], 20_000_000).unwrap();
        assert_eq!((plan.video.width, plan.video.height, plan.video.fps), (640, 480, 15));
        assert_eq!(plan.bytes_per_second, 18_432_000);
    }

    #[test]
    fn native_bgra_cannot_be_admitted_at_a_two_byte_rate() {
        wide_native_format_uses_its_larger_cost(*b"BGRA");
    }

    #[test]
    fn unknown_native_formats_use_a_conservative_four_byte_rate() {
        wide_native_format_uses_its_larger_cost(*b"ZZZZ");
    }

    #[test]
    fn mixed_native_formats_prefer_the_lower_cost_eligible_transport() {
        let mut wide = format(640, 480, 1, 30);
        wide.offered.mode.fourcc = *b"BGRA";
        let mut narrow = wide.clone();
        narrow.offered.mode.fourcc = *b"yuvs";
        let formats = [wide, narrow];
        for share in [20_000_000, u64::MAX] {
            let plan = plan_camera("cam", &VideoSettings::default(), &formats, share).unwrap();
            assert_eq!(plan.native.offered.mode.fourcc, *b"yuvs");
            assert_eq!(plan.video.fps, 30);
            assert_eq!(plan.bytes_per_second, 18_432_000);
        }
    }
    #[test]
    fn verified_faster_budget_keeps_720p_at_30() {
        let plan = plan_camera(
            "cam",
            &VideoSettings::default(),
            &inventory(),
            manifest_share(&cameras(3, Some(200_000_000))).unwrap(),
        )
        .unwrap();
        assert_eq!((plan.video.width, plan.video.height, plan.video.fps), (1280, 720, 30));
    }

    fn verified_hub_manifest() -> Manifest {
        let mut text = String::from(
            "[supervisor]\nmac_raw_budget_bytes_per_second=60000000\n[video]\nwidth=1280\nheight=720\nfps=10\n",
        );
        for index in 0..3 {
            text.push_str(&format!("[[device]]\nname='cam{index}'\nkind='camera'\nmatch={{name='Camera'}}\n"));
        }
        Manifest::parse(&text).unwrap()
    }

    #[test]
    fn verified_60_mb_budget_keeps_three_requested_720p_10_fps_cameras() {
        let manifest = verified_hub_manifest();
        let share = manifest_share(&manifest).unwrap();
        assert_eq!(share, 20_000_000);
        let mut total = 0;
        for camera in &manifest.devices {
            let plan = plan_camera(&camera.name, &camera.video().unwrap(), &inventory(), share).unwrap();
            assert_eq!((plan.video.width, plan.video.height, plan.video.fps), (1280, 720, 10));
            assert_eq!(plan.bytes_per_second, 18_432_000);
            total += plan.bytes_per_second;
        }
        assert_eq!(total, 55_296_000);
        assert!(total <= manifest.supervisor.mac_raw_budget_bytes_per_second);
    }

    #[test]
    fn verified_60_mb_budget_does_not_discount_unknown_four_byte_native_formats() {
        let manifest = verified_hub_manifest();
        let mut formats = inventory();
        for native in &mut formats {
            native.offered.mode.fourcc = *b"ZZZZ";
        }
        let mut total = 0;
        for camera in &manifest.devices {
            let plan =
                plan_camera(&camera.name, &camera.video().unwrap(), &formats, manifest_share(&manifest).unwrap())
                    .unwrap();
            assert_eq!((plan.video.width, plan.video.height, plan.video.fps), (640, 480, 10));
            assert_eq!(plan.bytes_per_second, 12_288_000);
            total += plan.bytes_per_second;
        }
        assert_eq!(total, 36_864_000);
    }

    #[test]
    fn same_width_smaller_actual_height_stays_within_the_admitted_native_cost() {
        let video = VideoSettings { width: 640, height: 480, fallback: vec![], ..VideoSettings::default() };
        for (fourcc, expected_fps) in [(CAPTURE_FOURCC, 30), (*b"ZZZZ", 15)] {
            let mut native = format(640, 480, 1, 30);
            native.offered.mode.fourcc = fourcc;
            let plan = plan_camera("cam", &video, &[native], 20_000_000).unwrap();
            assert_eq!(plan.video.fps, expected_fps);
            assert_eq!(plan.bytes_per_second, 18_432_000);
            let delivered_cost = raw_rate(plan.video.width, 360, plan.video.fps, native_pixel_bytes(fourcc)).unwrap();
            assert_eq!(delivered_cost, 13_824_000);
            assert!(delivered_cost <= plan.bytes_per_second);
            assert!(plan.bytes_per_second <= plan.share_bytes_per_second);
        }
    }
    #[test]
    fn ntsc_duration_is_valid_below_the_30_cap() {
        let mut native = format(640, 480, 30, 30);
        native.rates[0] =
            RateRange { min_fps: 29.97, max_fps: 29.97, min_duration: (1001, 30000), max_duration: (1001, 30000) };
        let plan = plan_camera("cam", &VideoSettings::default(), &[native], 20_000_000).unwrap();
        assert_eq!(plan.frame_duration, (1001, 30000));
        assert_eq!(plan.bytes_per_second, 18_432_000);
    }
    #[test]
    fn missing_inventory_or_smallest_unfitting_mode_is_actionable() {
        assert!(plan_camera("cam", &VideoSettings::default(), &[], 40_000_000).unwrap_err().contains("no offered"));
        assert!(plan_camera("cam", &VideoSettings::default(), &inventory(), 1).unwrap_err().contains("share"));
    }

    #[test]
    fn an_unoffered_or_overflowing_requested_mode_does_not_hide_a_safe_offered_fallback() {
        let video =
            VideoSettings { width: u32::MAX, height: u32::MAX, fallback: vec![(640, 480)], ..VideoSettings::default() };
        let plan = plan_camera("cam", &video, &inventory(), 13_333_333).unwrap();
        assert_eq!((plan.video.width, plan.video.height, plan.video.fps), (640, 480, 15));
    }

    #[test]
    fn auto_fit_does_not_upsize_a_small_request_to_a_default_fallback() {
        let video = VideoSettings { width: 320, height: 240, ..VideoSettings::default() };
        assert!(plan_camera("cam", &video, &[format(640, 480, 1, 30)], 40_000_000).is_err());
    }

    #[test]
    fn auto_fit_does_not_upsize_height_even_when_width_matches() {
        let video = VideoSettings { width: 640, height: 480, fallback: vec![(640, 720)], ..VideoSettings::default() };
        assert!(plan_camera("cam", &video, &[format(640, 720, 1, 30)], 40_000_000).is_err());
    }

    #[test]
    fn auto_fit_does_not_upsize_width_even_when_pixel_count_is_lower() {
        let video = VideoSettings { width: 640, height: 480, fallback: vec![(960, 240)], ..VideoSettings::default() };
        assert!(plan_camera("cam", &video, &[format(960, 240, 1, 30)], 40_000_000).is_err());
    }
    #[test]
    fn row_padding_overflow_and_exact_budget_boundary_are_checked() {
        assert_eq!(raw_rate(641, 480, 30, 2).unwrap(), 20_275_200);
        assert!(raw_rate(u32::MAX, u32::MAX, u32::MAX, 4).is_err());
        let video = VideoSettings { width: 640, height: 480, fps: 10, fallback: vec![], ..VideoSettings::default() };
        assert!(plan_camera("cam", &video, &inventory(), 6_144_000).is_ok());
        assert!(plan_camera("cam", &video, &inventory(), 6_143_999).is_err());
    }
    #[test]
    fn zero_budget_is_a_manifest_error_and_zero_share_is_refused() {
        assert!(Manifest::parse("[supervisor]\nmac_raw_budget_bytes_per_second=0").is_err());
        assert!(manifest_share(&cameras(3, Some(2))).is_err());
    }
}
