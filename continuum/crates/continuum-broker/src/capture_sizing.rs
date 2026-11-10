//! CLI-only projection of capture's pure TOML settings. No capture device libraries belong in the broker.
//! Defaults mirror `VideoSettings`: raw video keeps two seconds, MJPEG ten, with explicit
//! global/per-camera overrides winning. Compatible profile slots count toward camera capacity; only the shortfall
//! is added. New classes are introduced only when no profile class holds a frame.
//! Driver allocation bounds and stride padding can select a different class than the manifest estimate, so each
//! camera reserves capacity in every offered class, covering smaller JPEG buffers and larger raw strides.

use std::collections::BTreeSet;

use continuum_sched::SlotClass;
use serde::Deserialize;

pub const DEFAULT_CAMERA_BYTES: u64 = 4 << 30;
const PAGE: u64 = continuum_core::PAGE_ALIGN as u64;

#[derive(Default, Deserialize)]
struct Video {
    width: Option<u32>,
    height: Option<u32>,
    fps: Option<u32>,
    format: Option<String>,
    fallback: Option<Vec<(u32, u32)>>,
    ring_seconds: Option<f64>,
}

#[derive(Default, Deserialize)]
struct NetworkVideo {
    max_access_unit_bytes: Option<u32>,
    ring_bytes: Option<u64>,
}

#[derive(Deserialize)]
struct Device {
    name: String,
    kind: String,
    /// `[[device]].enabled = false` starts no capture worker, so the broker reserves no ring for it.
    #[serde(default = "enabled_by_default")]
    enabled: bool,
    #[serde(default)]
    video: Video,
    #[serde(default)]
    rtsp: Option<NetworkVideo>,
}

const fn enabled_by_default() -> bool {
    true
}

#[derive(Deserialize)]
struct Manifest {
    #[serde(default)]
    video: Video,
    #[serde(default, rename = "device")]
    devices: Vec<Device>,
}

#[derive(Debug)]
pub struct Plan {
    pub classes: Vec<SlotClass>,
    pub cameras: Vec<CameraPlan>,
    pub baseline_bytes: u64,
    /// Additional payload and slot-directory bytes, excluding the pre-existing profile and PCM reserve.
    pub camera_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Debug)]
pub struct CameraPlan {
    pub name: String,
    pub format: String,
    pub fps: u32,
    pub ring_seconds: f64,
    pub slots_per_class: u32,
    pub class_sizes: Vec<u64>,
    /// Encoded input has variable class counts instead of one uniform raw-frame reservation.
    pub network_class_slots: Option<Vec<(u64, u32)>>,
}

fn add(a: u64, b: u64) -> Result<u64, String> {
    a.checked_add(b).ok_or_else(|| "Tier 0 sizing arithmetic overflow".into())
}

fn mul(a: u64, b: u64) -> Result<u64, String> {
    a.checked_mul(b).ok_or_else(|| "Tier 0 sizing arithmetic overflow".into())
}

fn align(value: u64, by: u64) -> Result<u64, String> {
    Ok(add(value, by - 1)? / by * by)
}

/// Total slab layout, including the documented 64-byte slot metadata, 16 KiB superblock, and PCM ring reserve.
fn total_bytes(classes: &[SlotClass], ring_bytes: u64) -> Result<u64, String> {
    let mut metadata = PAGE;
    let mut payload = 0;
    for class in classes {
        if class.count == 0 || class.size_bytes == 0 {
            return Err("Tier 0 classes need nonzero sizes and counts".into());
        }
        metadata = add(metadata, mul(u64::from(class.count), 64)?)?;
        payload = add(payload, mul(align(class.size_bytes, PAGE)?, u64::from(class.count))?)?;
    }
    add(add(align(metadata, PAGE)?, payload)?, align(ring_bytes, PAGE)?)
}

/// A camera frame's planned bytes, shared with capture's class pick (`continuum_core::camera_frame_bound`).
/// AVFoundation delivers UYVY even if MJPEG was requested, so a Mac's frames are raw.
fn frame_bytes(width: u32, height: u32, format: &str, mac: bool) -> Result<u64, String> {
    if width == 0 || height == 0 {
        return Err("camera width and height must be positive".into());
    }
    continuum_core::camera_frame_bound(width, height, mac || format == "UYVY")
        .ok_or_else(|| "Tier 0 sizing arithmetic overflow".into())
}

pub fn plan(text: &str, baseline: &[SlotClass], mac: bool, budget: u64, ring_bytes: u64) -> Result<Plan, String> {
    let manifest: Manifest = toml::from_str(text).map_err(|error: toml::de::Error| {
        let location = error.span().map_or(String::new(), |span| format!(" near byte {}", span.start));
        format!("invalid capture manifest{location}; source values omitted")
    })?;
    let mut classes = baseline.to_vec();
    let mut requirements = Vec::new();
    let mut usb_bounds: Vec<Vec<u64>> = Vec::new(); // each USB camera's planned frame sizes, in requirements' order
    let mut networks = Vec::new();
    let mut names = BTreeSet::new();
    for device in manifest.devices {
        if device.name.is_empty()
            || !device.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            || !names.insert(device.name.clone())
        {
            return Err(format!("invalid or duplicate capture device name {:?}", device.name));
        }
        if !matches!(device.kind.as_str(), "camera" | "rtsp" | "microphone" | "xvf3800" | "teensy") {
            return Err(format!("{}: unknown capture device kind {:?}", device.name, device.kind));
        }
        // A disabled row keeps its name (still unique, so its stream identity and history stay), but nothing
        // captures it until it is enabled again and the broker restarts with a new reservation.
        if !device.enabled {
            continue;
        }
        if device.kind == "rtsp" {
            networks.push(device);
            continue;
        }
        if device.kind != "camera" {
            continue;
        }
        let own = device.video;
        let global = &manifest.video;
        let width = own.width.or(global.width).unwrap_or(1280);
        let height = own.height.or(global.height).unwrap_or(720);
        let fps = own.fps.or(global.fps).unwrap_or(30);
        let format = own.format.as_deref().or(global.format.as_deref()).unwrap_or("MJPEG");
        let seconds =
            own.ring_seconds.or(global.ring_seconds).unwrap_or(if mac || format == "UYVY" { 2.0 } else { 10.0 });
        if !matches!(format, "MJPEG" | "UYVY") || fps == 0 || !seconds.is_finite() || seconds <= 0.0 {
            return Err(format!(
                "{}: video requires MJPEG/UYVY, positive fps and finite positive ring_seconds",
                device.name
            ));
        }
        let mut bounds = vec![frame_bytes(width, height, format, mac)?];
        let defaults = vec![(960, 540), (640, 480)];
        for &(w, h) in own.fallback.as_ref().or(global.fallback.as_ref()).unwrap_or(&defaults) {
            bounds.push(frame_bytes(w, h, format, mac)?);
        }
        let frames = (f64::from(fps) * seconds).ceil();
        if !frames.is_finite() || frames > f64::from(u32::MAX) || frames < 1.0 {
            return Err(format!("{}: requested video retention has too many frames", device.name));
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // bounded finite integral value above
        let retained = frames as u32;
        let count = add(add(u64::from(retained), u64::from(retained).div_ceil(4))?, 8)?;
        let count = u32::try_from(count).map_err(|_| format!("{}: camera slot count overflow", device.name))?;
        for &bound in &bounds {
            let size = bound.max(PAGE).checked_next_power_of_two().ok_or("camera frame size overflow")?;
            if !baseline.iter().any(|class| class.size_bytes >= bound)
                && !classes.iter().any(|class| class.size_bytes == size)
            {
                classes.push(SlotClass { size_bytes: size, count: 0 });
            }
        }
        usb_bounds.push(bounds);
        requirements.push(CameraPlan {
            name: device.name,
            format: if mac { "UYVY".into() } else { format.into() },
            fps,
            ring_seconds: seconds,
            slots_per_class: count,
            class_sizes: Vec::new(),
            network_class_slots: None,
        });
    }
    let mut cameras = Vec::new();
    let mut demand = vec![0u32; classes.len()];
    for mut camera in requirements {
        // Count each camera once per offered class after all classes have been introduced. Never credit an
        // estimated class while the actual driver allocation can select another one.
        for (index, class) in classes.iter_mut().enumerate() {
            demand[index] =
                demand[index].checked_add(camera.slots_per_class).ok_or("combined camera slot count overflow")?;
            class.count = class.count.max(demand[index]);
            camera.class_sizes.push(class.size_bytes);
        }
        cameras.push(camera);
    }
    // Add encoded reservations after ordinary cameras: network-only classes must not multiply USB demand.
    let usb_classes = classes.len();
    for device in networks {
        let settings = device.rtsp.ok_or_else(|| format!("{}: rtsp settings are required", device.name))?;
        let maximum = u64::from(settings.max_access_unit_bytes.unwrap_or(4 << 20));
        let ring_bytes = settings.ring_bytes.unwrap_or(64 << 20);
        let seconds = device.video.ring_seconds.unwrap_or(10.0);
        let fps = device.video.fps.unwrap_or(30);
        if !(64 << 10..=16 << 20).contains(&maximum)
            || !(1..=240).contains(&fps)
            || !seconds.is_finite()
            || seconds <= 0.0
            || seconds > 3600.0
            || ring_bytes > 1 << 30
        {
            return Err(format!("{}: invalid RTSP encoded size/rate/time/byte budget", device.name));
        }
        let largest = maximum.max(PAGE).checked_next_power_of_two().ok_or("RTSP class size overflow")?;
        let maximum_cost = add(largest, 64)?;
        let minimum = mul(2, maximum_cost)?;
        if ring_bytes < minimum {
            return Err(format!(
                "{}: RTSP ring_bytes must hold two maximum access units including slot metadata",
                device.name
            ));
        }
        let mut size = PAGE;
        loop {
            if !classes.iter().any(|class| class.size_bytes == size) {
                classes.push(SlotClass { size_bytes: size, count: 0 });
                demand.push(0);
            }
            if size == largest {
                break;
            }
            size = size.saturating_mul(4).min(largest);
        }
        let eligible: Vec<usize> = classes
            .iter()
            .enumerate()
            .filter_map(|(index, class)| (class.size_bytes <= largest).then_some(index))
            .collect();
        let smaller = eligible.iter().filter(|&&index| classes[index].size_bytes < largest).count();
        let share = (ring_bytes - minimum) / u64::try_from(smaller).unwrap_or(1).max(1);
        let target = (f64::from(fps) * seconds).ceil() as u64;
        let target = add(add(target, target.div_ceil(4))?, 8)?;
        let mut network_class_slots = Vec::new();
        for index in eligible {
            let size = classes[index].size_bytes;
            let count = if size == largest { 2 } else { (share / add(size, 64)?).min(target) };
            let count = u32::try_from(count).map_err(|_| "RTSP slot count overflow")?;
            if count == 0 {
                continue;
            }
            demand[index] = demand[index].checked_add(count).ok_or("combined RTSP slot count overflow")?;
            classes[index].count = classes[index].count.max(demand[index]);
            network_class_slots.push((size, count));
        }
        cameras.push(CameraPlan {
            name: device.name,
            format: "H264AnnexB".into(),
            fps,
            ring_seconds: seconds,
            // The uniform count is unused when network_class_slots is present.
            slots_per_class: 2,
            class_sizes: network_class_slots.iter().map(|&(size, _)| size).collect(),
            network_class_slots: Some(network_class_slots),
        });
    }
    // A USB camera claims the smallest class that holds its planned frame (capture's pick_class over
    // camera_frame_bound). Where that's a class an RTSP camera added above, its slots were never reserved there, so
    // it shared the RTSP camera's few and dropped frames: reserve them in that one class too.
    for (camera, bounds) in cameras.iter_mut().zip(&usb_bounds) {
        for &bound in bounds {
            let claimed = classes
                .iter()
                .enumerate()
                .filter(|(_, class)| class.size_bytes >= bound)
                .min_by_key(|(_, class)| class.size_bytes)
                .map(|(index, _)| index);
            let Some(index) = claimed.filter(|&index| index >= usb_classes) else { continue };
            if camera.class_sizes.contains(&classes[index].size_bytes) {
                continue; // two of its sizes land in the same class: one reservation
            }
            demand[index] =
                demand[index].checked_add(camera.slots_per_class).ok_or("combined camera slot count overflow")?;
            classes[index].count = classes[index].count.max(demand[index]);
            camera.class_sizes.push(classes[index].size_bytes);
        }
    }
    // Only nonempty offered classes belong in the arena.
    classes.retain(|class| class.count != 0);
    if classes.len() > continuum_shm::arena::MAX_CLASSES {
        return Err("capture manifest needs more than 16 Tier 0 size classes; use --tier0-classes explicitly".into());
    }
    let total_bytes = total_bytes(&classes, ring_bytes)?;
    let baseline_bytes = self::total_bytes(baseline, ring_bytes)?;
    let camera_bytes = total_bytes.checked_sub(baseline_bytes).ok_or("camera plan reduced profile capacity")?;
    if camera_bytes > budget {
        return Err(format!(
            "camera Tier 0 reservation needs {camera_bytes} bytes, exceeding --tier0-camera-mib budget {budget}; reduce ring_seconds/fps/resolution or explicitly raise the budget"
        ));
    }
    usize::try_from(total_bytes).map_err(|_| "Tier 0 arena exceeds this platform's address space")?;
    Ok(Plan { classes, cameras, baseline_bytes, camera_bytes, total_bytes })
}

/// How many times the profile's headroom may be halved to fit a small `/dev/shm`: at 10 the Linux profile's 11 GB
/// is under 20 MB, which a container's default 64 MB holds.
pub const MAX_SHRINK: u32 = 10;
/// The ring region is never shrunk below this: the memory log's and the demo's rings fit in it.
const MIN_RING_BYTES: u64 = 1 << 20;

/// A plan chosen to fit the shared memory there is.
#[derive(Debug)]
pub struct Fit {
    pub plan: Plan,
    /// The profile's slot counts were halved this many times (0: the whole profile).
    pub shrink: u32,
    /// The ring region the plan was laid out with: the caller's, or less when nothing captures and it was shrunk.
    pub ring_bytes: u64,
}

/// The profile's classes with each count halved `shrink` times, never below one slot: the sizes and their order
/// stay, since a stream names its class by index.
fn shrunk(baseline: &[SlotClass], shrink: u32) -> Vec<SlotClass> {
    baseline
        .iter()
        .map(|class| SlotClass { size_bytes: class.size_bytes, count: (class.count >> shrink).max(1) })
        .collect()
}

/// Whether the manifest has a device that captures: only then is the ring region (PCM, IMU) kept whole.
fn captures(text: &str) -> bool {
    toml::from_str::<Manifest>(text).is_ok_and(|manifest| manifest.devices.iter().any(|device| device.enabled))
}

/// The plan for this manifest that the shared memory can hold. The configured devices always get what their
/// settings ask for (`plan`); what shrinks is the profile's headroom beyond them. The whole profile is kept when
/// the arena then takes no more than half of `room` (the rest of `/dev/shm` is see-decode's frame ring and other
/// programs'), else it is halved until it does, up to `MAX_SHRINK` times. With nothing capturing, the ring region
/// shrinks with it. When even the smallest plan is larger than all of `room`, the error names what to change.
/// `room` is `None` where shared memory has no size to fit (macOS): the whole profile, as before.
pub fn fit(
    text: &str,
    baseline: &[SlotClass],
    mac: bool,
    budget: u64,
    ring_bytes: u64,
    room: Option<u64>,
) -> Result<Fit, String> {
    // The whole profile first: it validates the manifest and holds the camera budget to what it always meant,
    // the bytes cameras add to the profile.
    let whole = plan(text, baseline, mac, budget, ring_bytes)?;
    let Some(room) = room else { return Ok(Fit { plan: whole, shrink: 0, ring_bytes }) };
    if whole.total_bytes <= room / 2 {
        return Ok(Fit { plan: whole, shrink: 0, ring_bytes });
    }
    let idle = !captures(text);
    let mut smallest = None;
    for shrink in 1..=MAX_SHRINK {
        let ring = if idle { (ring_bytes >> shrink).max(MIN_RING_BYTES.min(ring_bytes)) } else { ring_bytes };
        let candidate = plan(text, &shrunk(baseline, shrink), mac, u64::MAX, ring)?;
        if candidate.total_bytes <= room / 2 {
            return Ok(Fit { plan: candidate, shrink, ring_bytes: ring });
        }
        smallest = Some(Fit { plan: candidate, shrink, ring_bytes: ring });
    }
    let smallest = smallest.ok_or("no Tier 0 plan")?;
    if smallest.plan.total_bytes <= room {
        return Ok(smallest);
    }
    let needs = smallest.plan.total_bytes;
    Err(if smallest.plan.cameras.is_empty() {
        format!(
            "the Tier 0 arena needs at least {needs} bytes but /dev/shm holds {room}: give /dev/shm more room (a \
             container: --shm-size)"
        )
    } else {
        format!(
            "the Tier 0 arena needs {needs} bytes for the {} cameras in the device manifest but /dev/shm holds \
             {room}: lower ring_seconds, fps, or the frame size under [video] (or a camera's own video settings) \
             there, turn a camera off with enabled = false, or give /dev/shm more room",
            smallest.plan.cameras.len()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn raw(cameras: usize, extra: &str) -> String {
        let mut text = format!("[video]\nwidth=1920\nheight=1080\nfps=30\nformat='UYVY'\n{extra}\n");
        for index in 0..cameras {
            text.push_str(&format!("[[device]]\nname='cam{index}'\nkind='camera'\nmatch={{name='Camera'}}\n"));
        }
        text
    }

    #[test]
    fn three_raw_1080p_cameras_get_two_seconds_plus_headroom_without_large_allocations() {
        let baseline = continuum_sched::Profile::mac_mini_64gb().slot_classes;
        let plan = plan(&raw(3, ""), &baseline, true, DEFAULT_CAMERA_BYTES, 64 << 20).unwrap();
        assert_eq!(plan.classes, baseline);
        assert_eq!(plan.cameras[0].slots_per_class, 83); // 60 frames + 15 extra + 8, or 249 across three.
        assert_eq!(plan.cameras[0].class_sizes, vec![64 << 10, 2 << 20, 16 << 20]);
        assert_eq!(plan.camera_bytes, 0);
        assert!(
            !plan.classes.iter().any(|c| c.size_bytes == 4 << 20),
            "a new class would invalidate profile capacity credit"
        );
        assert_eq!(plan.cameras.len(), 3);
        assert!(plan.total_bytes > plan.camera_bytes);
    }

    #[test]
    fn collisions_extend_only_the_shortfall_in_the_smallest_final_class() {
        let base = [SlotClass { size_bytes: 4 << 20, count: 7 }];
        let plan = plan(&raw(2, "fallback=[]"), &base, true, DEFAULT_CAMERA_BYTES, 0).unwrap();
        assert_eq!(plan.classes, vec![SlotClass { size_bytes: 4 << 20, count: 166 }]);
    }

    #[test]
    fn per_camera_overrides_and_larger_fallbacks_determine_capacity() {
        let text = "[video]\nwidth=640\nheight=480\nfps=15\nring_seconds=1\n[[device]]\nname='cam'\nkind='camera'\nvideo={fps=30,ring_seconds=2,fallback=[[1920,1080]]}";
        let plan = plan(text, &[], true, DEFAULT_CAMERA_BYTES, 0).unwrap();
        assert_eq!(
            plan.classes,
            vec![SlotClass { size_bytes: 1 << 20, count: 83 }, SlotClass { size_bytes: 4 << 20, count: 83 }]
        );
    }

    #[test]
    fn mac_mjpeg_requests_still_reserve_padded_raw_frames() {
        assert_eq!(frame_bytes(1921, 1080, "MJPEG", true).unwrap(), 3904 * 1080);
        assert_eq!(frame_bytes(1920, 1080, "MJPEG", false).unwrap(), 4_147_200 + 65_536);
    }

    fn actual_driver_capacity(format: &str, width: u32, height: u32, actual_bytes: u64, expected_class: u64) {
        let base = [SlotClass { size_bytes: 2 << 20, count: 1 }, SlotClass { size_bytes: 16 << 20, count: 1 }];
        let text = format!(
            "[[device]]\nname='cam'\nkind='camera'\nvideo={{format='{format}',width={width},height={height},fps=30,fallback=[]}}"
        );
        let plan = plan(&text, &base, false, 16 << 30, 0).unwrap();
        let actual = plan.classes.iter().filter(|c| c.size_bytes >= actual_bytes).min_by_key(|c| c.size_bytes).unwrap();
        assert_eq!(actual.size_bytes, expected_class);
        assert!(
            actual.count >= plan.cameras[0].slots_per_class,
            "{format}: actual driver buffer {actual_bytes} selects an underprovisioned class"
        );
    }

    #[test]
    fn actual_driver_allocation_can_choose_a_smaller_class_than_the_estimate() {
        actual_driver_capacity("MJPEG", 1920, 1080, 2 << 20, 2 << 20);
    }

    #[test]
    fn actual_driver_allocation_can_choose_a_larger_class_than_the_estimate() {
        actual_driver_capacity("UYVY", 1928, 536, 2_126_848, 16 << 20);
    }

    #[test]
    fn compressed_retention_stays_ten_seconds_while_native_raw_defaults_to_two() {
        let text = "[video]\nwidth=1920\nheight=1080\nfps=30\nfallback=[]\n[[device]]\nname='compressed'\nkind='camera'\n[[device]]\nname='raw'\nkind='camera'\nvideo={format='UYVY'}";
        let linux = plan(text, &[], false, 8 << 30, 0).unwrap();
        assert_eq!((linux.cameras[0].ring_seconds, linux.cameras[0].slots_per_class), (10.0, 383));
        assert_eq!((linux.cameras[1].ring_seconds, linux.cameras[1].slots_per_class), (2.0, 83));
        let mac = plan(text, &[], true, DEFAULT_CAMERA_BYTES, 0).unwrap();
        assert_eq!((mac.cameras[0].format.as_str(), mac.cameras[0].ring_seconds), ("UYVY", 2.0));
        assert_eq!(mac.classes[0].count, 166);
    }

    #[test]
    fn explicit_global_retention_and_camera_override_win_over_raw_defaults() {
        let text = raw(1, "ring_seconds=10\nfallback=[]");
        assert_eq!(plan(&text, &[], true, DEFAULT_CAMERA_BYTES, 0).unwrap().cameras[0].ring_seconds, 10.0);
        let text = format!("{text}video={{ring_seconds=3}}\n");
        assert_eq!(plan(&text, &[], true, DEFAULT_CAMERA_BYTES, 0).unwrap().cameras[0].slots_per_class, 121);
    }

    #[test]
    fn fallback_modes_share_one_reservation_when_their_final_class_matches() {
        let text = "[[device]]\nname='cam'\nkind='camera'\nvideo={width=640,height=480,format='UYVY',fallback=[[640,480],[320,240]]}";
        let base = [SlotClass { size_bytes: 2 << 20, count: 10 }];
        let plan = plan(text, &base, true, DEFAULT_CAMERA_BYTES, 0).unwrap();
        assert_eq!(plan.cameras[0].class_sizes, vec![2 << 20]);
        assert_eq!(plan.classes, vec![SlotClass { size_bytes: 2 << 20, count: 83 }]);
    }

    #[test]
    fn class_limit_and_combined_slot_overflow_fail_before_any_allocation() {
        let base: Vec<SlotClass> = (1..=16).map(|n| SlotClass { size_bytes: n * PAGE, count: 1 }).collect();
        assert!(plan(&raw(1, "fallback=[]"), &base, true, DEFAULT_CAMERA_BYTES, 0).unwrap_err().contains("16"));
        let text = "[video]\nfps=4294967295\nring_seconds=1\n[[device]]\nname='cam'\nkind='camera'";
        assert!(plan(text, &[], true, DEFAULT_CAMERA_BYTES, 0).unwrap_err().contains("count overflow"));
    }

    #[test]
    fn explicit_retention_is_honored_or_rejected_never_clamped() {
        let text = raw(3, "ring_seconds=10");
        assert!(plan(&text, &[], true, DEFAULT_CAMERA_BYTES, 0).unwrap_err().contains("budget"));
        let plan = plan(&text, &[], true, 8 << 30, 0).unwrap();
        assert_eq!(plan.classes[0].count, 1149);
    }

    #[test]
    fn invalid_and_overflowing_requirements_are_actionable_errors() {
        for extra in ["ring_seconds=nan", "ring_seconds=0", "ring_seconds=1e100", "ring_seconds=-1"] {
            assert!(plan(&raw(1, extra), &[], true, DEFAULT_CAMERA_BYTES, 0).is_err());
        }
        for text in [
            "[broken",
            "device='not an array'",
            "[[device]]\nname='cam'\nkind='camera'\nvideo={width=0}",
            "[[device]]\nname='cam'\nkind='camera'\nvideo={width=4294967295,height=4294967295}",
        ] {
            assert!(plan(text, &[], true, DEFAULT_CAMERA_BYTES, 0).is_err());
        }
        assert!(total_bytes(&[SlotClass { size_bytes: u64::MAX, count: u32::MAX }], 0).is_err());
    }

    #[test]
    fn a_bounded_real_arena_uses_the_planned_classes_and_accounted_layout() {
        let text = "[[device]]\nname='cam'\nkind='camera'\nvideo={width=64,height=48,fps=1,ring_seconds=1,fallback=[]}";
        let plan = plan(text, &[], true, DEFAULT_CAMERA_BYTES, 16 << 10).unwrap();
        assert!(plan.total_bytes < 256 << 10);
        let dir = tempfile::tempdir().unwrap();
        let arena =
            continuum_broker::tier0::Tier0::create(&dir.path().join("broker.sock"), &plan.classes, 16 << 10).unwrap();
        assert_eq!(arena.summary().size_bytes, plan.total_bytes);
        assert_eq!(arena.summary().classes, vec![(16 << 10, 10)]);
    }
}

#[cfg(test)]
mod network_tests {
    use super::tests::raw;
    use super::*;

    #[test]
    fn a_disabled_camera_reserves_nothing_and_enabling_it_reserves_its_ring_again() {
        let base = [SlotClass { size_bytes: 4 << 20, count: 7 }];
        let one = plan(&raw(1, "fallback=[]"), &base, true, DEFAULT_CAMERA_BYTES, 0).unwrap();
        let enabled = raw(2, "fallback=[]").replace("name='cam1'", "name='cam1'\nenabled=true");
        let disabled = enabled.replace("enabled=true", "enabled=false");
        let both = plan(&enabled, &base, true, DEFAULT_CAMERA_BYTES, 0).unwrap();
        let off = plan(&disabled, &base, true, DEFAULT_CAMERA_BYTES, 0).unwrap();
        assert_eq!(off.cameras.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), ["cam0"]);
        assert_eq!(off.classes, one.classes, "a disabled camera adds no slots");
        assert_eq!(both.cameras.len(), 2);
        assert_eq!(both.classes, plan(&raw(2, "fallback=[]"), &base, true, DEFAULT_CAMERA_BYTES, 0).unwrap().classes);
        assert!(both.classes[0].count > off.classes[0].count);
    }

    #[test]
    fn a_disabled_rtsp_camera_reserves_nothing_but_its_name_stays_unique() {
        let on = plan(&rtsp("door"), &[], false, DEFAULT_CAMERA_BYTES, 0).unwrap();
        let off = plan(
            &rtsp("door").replace("kind='rtsp'", "kind='rtsp'\nenabled=false"),
            &[],
            false,
            DEFAULT_CAMERA_BYTES,
            0,
        )
        .unwrap();
        assert!(on.camera_bytes > 0);
        assert_eq!((off.camera_bytes, off.cameras.len()), (0, 0));
        let twice = format!("{}{}", rtsp("door").replace("kind='rtsp'", "kind='rtsp'\nenabled=false"), rtsp("door"));
        assert!(plan(&twice, &[], false, DEFAULT_CAMERA_BYTES, 0).unwrap_err().contains("duplicate"));
    }

    #[test]
    fn a_usb_camera_has_its_slots_in_every_class_it_claims_including_one_an_rtsp_camera_added() {
        // A USB camera (cam0: 960x540 MJPEG, falling back to 640x480) beside an RTSP camera, whose classes include
        // 1 MiB. cam0 claims the smallest class that holds each planned frame (capture's pick over
        // continuum_core::camera_frame_bound): 2 MiB for 960 x 2 x 540 + 64 KiB = 1,102,336 bytes, and the RTSP
        // camera's 1 MiB for 640 x 2 x 480 + 64 KiB = 679,936. Its 30 x 10 s + 25% + 8 = 383 slots go in both.
        let base = [SlotClass { size_bytes: 64 << 10, count: 64 }, SlotClass { size_bytes: 2 << 20, count: 64 }];
        let usb = "[[device]]\nname='cam0'\nkind='camera'\nmatch={usb='0fd9:0078'}\n[device.video]\nwidth=960\nheight=540\nfps=30\n";
        let both = plan(&format!("{usb}{}", rtsp("patio")), &base, false, DEFAULT_CAMERA_BYTES, 64 << 20).unwrap();
        let alone = plan(&rtsp("patio"), &base, false, DEFAULT_CAMERA_BYTES, 64 << 20).unwrap();
        let cam0 = &both.cameras[0];
        assert_eq!(cam0.slots_per_class, 383);
        for (bound, claimed) in [(1_102_336_u64, 2_u64 << 20), (679_936, 1 << 20)] {
            let smallest = both.classes.iter().filter(|c| c.size_bytes >= bound).map(|c| c.size_bytes).min();
            assert_eq!(smallest, Some(claimed), "the class capture claims for {bound} bytes");
            assert!(cam0.class_sizes.contains(&claimed), "cam0 has slots in its {claimed} byte class");
        }
        let count = |plan: &Plan, size: u64| plan.classes.iter().find(|c| c.size_bytes == size).map_or(0, |c| c.count);
        assert_eq!(count(&both, 1 << 20), count(&alone, 1 << 20) + 383, "the RTSP camera's own slots stay its own");
    }

    fn rtsp(name: &str) -> String {
        format!(
            "[[device]]\nname='{name}'\nkind='rtsp'\nrtsp={{url='rtsps://user:private-password@camera.invalid/private-token',ring_bytes=67108864}}\n"
        )
    }

    #[test]
    fn encoded_reservations_follow_a_byte_budget_instead_of_uncompressed_geometry_or_max_au_times_frames() {
        let camera = plan(&rtsp("entryway"), &[], false, DEFAULT_CAMERA_BYTES, 0).unwrap();
        let network = &camera.cameras[0];
        let classes = network.network_class_slots.as_ref().unwrap();
        assert!(classes.contains(&(4 << 20, 2)));
        let held: u64 = classes.iter().map(|&(size, count)| (size + 64) * u64::from(count)).sum();
        assert!(held <= 64 << 20);
        assert!(camera.camera_bytes < 65 << 20);
        assert_eq!(network.format, "H264AnnexB");
        assert_eq!(
            camera.classes,
            plan(&rtsp("entryway"), &[], true, DEFAULT_CAMERA_BYTES, 0).unwrap().classes,
            "macOS must not turn encoded input into raw UYVY sizing"
        );
        assert!(
            plan(&(rtsp("entryway") + &rtsp("workroom")), &[], false, 64 << 20, 0).is_err(),
            "global camera budget still applies"
        );
    }

    #[test]
    fn network_classes_do_not_multiply_existing_usb_reservations_and_errors_hide_url_values() {
        let usb =
            "[[device]]\nname='usb'\nkind='camera'\nvideo={width=640,height=480,fps=15,ring_seconds=2,fallback=[]}\n";
        let before = plan(usb, &[], false, DEFAULT_CAMERA_BYTES, 0).unwrap();
        let after = plan(&(String::from(usb) + &rtsp("entryway")), &[], false, DEFAULT_CAMERA_BYTES, 0).unwrap();
        assert_eq!(before.cameras[0].class_sizes, after.cameras[0].class_sizes);
        assert_eq!(before.cameras[0].slots_per_class, after.cameras[0].slots_per_class);
        let malformed =
            rtsp("entryway").replace("ring_bytes=67108864", "ring_bytes='rtsps://private-password/private-token'");
        let error = plan(&malformed, &[], false, DEFAULT_CAMERA_BYTES, 0).unwrap_err();
        assert!(!error.contains("private-password"));
        assert!(!error.contains("private-token"));
    }
}

#[cfg(test)]
mod fit_tests {
    use super::tests::raw;
    use super::*;

    const GIB: u64 = 1 << 30;
    const RING: u64 = 64 << 20;

    fn linux() -> Vec<SlotClass> {
        continuum_sched::Profile::linux_dual_3090().slot_classes
    }

    fn mjpeg(cameras: usize) -> String {
        let mut text = String::new();
        for index in 0..cameras {
            text.push_str(&format!("[[device]]\nname='cam{index}'\nkind='camera'\n"));
        }
        text
    }

    #[test]
    fn a_machine_with_room_keeps_the_whole_profile_exactly_as_before() {
        for text in [String::new(), mjpeg(1), raw(2, "")] {
            let before = plan(&text, &linux(), false, DEFAULT_CAMERA_BYTES, RING).unwrap();
            for room in [None, Some(before.total_bytes * 2), Some(64 * GIB)] {
                let fit = fit(&text, &linux(), false, DEFAULT_CAMERA_BYTES, RING, room).unwrap();
                assert_eq!((fit.shrink, fit.ring_bytes), (0, RING));
                assert_eq!(fit.plan.classes, before.classes);
                assert_eq!(fit.plan.total_bytes, before.total_bytes);
            }
        }
    }

    #[test]
    fn with_no_devices_the_headroom_and_the_ring_region_shrink_until_half_of_shm_holds_them() {
        // Docker's default /dev/shm (64 MiB), a small machine (2 GiB of it), and a middling one (16 GiB).
        for (room, shrink) in [(64 << 20, 9), (2 * GIB, 4), (16 * GIB, 1)] {
            let fit = fit("", &linux(), false, DEFAULT_CAMERA_BYTES, RING, Some(room)).unwrap();
            assert_eq!(fit.shrink, shrink, "room {room}");
            assert!(fit.plan.total_bytes <= room / 2);
            // One step less would not have fit: it is the most headroom that does.
            let larger = plan("", &shrunk(&linux(), shrink - 1), false, u64::MAX, RING >> (shrink - 1)).unwrap();
            assert!(larger.total_bytes > room / 2);
            // The classes keep their sizes and order (a stream names its class by index), and none is empty.
            assert_eq!(
                fit.plan.classes.iter().map(|c| c.size_bytes).collect::<Vec<_>>(),
                linux().iter().map(|c| c.size_bytes).collect::<Vec<_>>()
            );
            assert!(fit.plan.classes.iter().all(|c| c.count >= 1));
        }
    }

    #[test]
    fn configured_cameras_keep_every_slot_their_settings_ask_for_whatever_the_headroom() {
        let text = "[[device]]\nname='cam'\nkind='camera'\nvideo={ring_seconds=1}"; // 30 + 8 + 8 = 46 slots a class
        let whole = plan(text, &linux(), false, DEFAULT_CAMERA_BYTES, RING).unwrap();
        let fit = fit(text, &linux(), false, DEFAULT_CAMERA_BYTES, RING, Some(4 * GIB)).unwrap();
        assert!(fit.shrink > 0 && fit.plan.total_bytes <= 2 * GIB);
        assert_eq!(fit.ring_bytes, RING, "a capturing install keeps its whole ring region");
        assert_eq!(fit.plan.cameras[0].slots_per_class, whole.cameras[0].slots_per_class);
        for size in &fit.plan.cameras[0].class_sizes {
            let class = fit.plan.classes.iter().find(|c| c.size_bytes == *size).unwrap();
            assert!(class.count >= 46, "class {size} holds {} slots", class.count);
        }
    }

    #[test]
    fn more_than_half_but_within_shm_still_starts_on_the_smallest_headroom() {
        let text = "[[device]]\nname='cam'\nkind='camera'\nvideo={ring_seconds=1}";
        let smallest = plan(text, &shrunk(&linux(), MAX_SHRINK), false, u64::MAX, RING).unwrap().total_bytes;
        let fit = fit(text, &linux(), false, DEFAULT_CAMERA_BYTES, RING, Some(smallest + 1)).unwrap();
        assert_eq!((fit.shrink, fit.plan.total_bytes), (MAX_SHRINK, smallest));
    }

    #[test]
    fn cameras_that_cannot_fit_say_which_settings_to_lower_and_how_much_is_needed() {
        // One MJPEG camera at its default ten seconds reserves 383 slots in the 16 MiB class alone: about 6 GiB.
        let error = fit(&mjpeg(1), &linux(), false, DEFAULT_CAMERA_BYTES, RING, Some(4 * GIB)).unwrap_err();
        assert!(error.contains("for the 1 cameras in the device manifest"), "{error}");
        assert!(error.contains("ring_seconds") && error.contains("enabled = false"), "{error}");
        assert!(error.contains(&format!("/dev/shm holds {}", 4 * GIB)), "{error}");
        let none = fit("", &linux(), false, DEFAULT_CAMERA_BYTES, RING, Some(8 << 20)).unwrap_err();
        assert!(none.contains("--shm-size") && !none.contains("cameras"), "{none}");
    }

    #[test]
    fn the_camera_budget_still_means_bytes_added_to_the_whole_profile() {
        // Within budget against the whole profile, so shrinking the headroom must not make it an error.
        assert!(fit(&mjpeg(1), &linux(), false, DEFAULT_CAMERA_BYTES, RING, Some(20 * GIB)).is_ok());
        // And a manifest over the budget is refused as before, room or not.
        let over = raw(40, "ring_seconds=30");
        assert!(plan(&over, &linux(), false, DEFAULT_CAMERA_BYTES, RING).unwrap_err().contains("--tier0-camera-mib"));
        let error = fit(&over, &linux(), false, DEFAULT_CAMERA_BYTES, RING, Some(GIB)).unwrap_err();
        assert!(error.contains("--tier0-camera-mib"), "{error}");
    }

    #[test]
    fn a_disabled_device_does_not_keep_the_ring_region_whole() {
        let off = "[[device]]\nname='mic'\nkind='microphone'\nenabled=false";
        let on = "[[device]]\nname='mic'\nkind='microphone'";
        assert!(fit(off, &linux(), false, DEFAULT_CAMERA_BYTES, RING, Some(GIB)).unwrap().ring_bytes < RING);
        assert_eq!(fit(on, &linux(), false, DEFAULT_CAMERA_BYTES, RING, Some(GIB)).unwrap().ring_bytes, RING);
    }
}
