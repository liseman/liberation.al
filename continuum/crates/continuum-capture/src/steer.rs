//! A camera's live controls and auto exposure, running beside its capture loop: the loop offers a frame a second
//! ([`Steer::offer`], a copy, never waiting), and a thread of its own measures it and steers the camera
//! ([`crate::exposure::AutoExposure`]), so a decode never costs the 30 fps loop a frame.
//!
//! - **What it measures:** the frame's luma ([`crate::exposure::luma_plane`]), over the faces in view when the face
//!   worker has seen some in the last second (the `face.boxes` ring in Tier 0, about 50 to 150 ms behind the frame),
//!   else over the whole frame. So a face in front of a bright window is what gets exposed.
//! - **What it sets (V4L2):** `exposure_time_absolute`, with the camera in shutter-priority exposure, where the
//!   camera raises its own gain as the exposure is capped at the frame period (the Elgato Facecam has no gain
//!   control of its own), else its brightness. A camera with neither is put in its own auto mode instead, said once
//!   at WARN, so its picture still adapts, at whatever frame rate the camera picks.
//! - **macOS:** AVFoundation's custom exposure (duration and ISO) is still to come; the camera keeps its own
//!   auto exposure there.
//!
//! The same thread polls manifest preferences once per second and applies saved controls, even with steering OFF.
//! Mode changes and manual writes share this owner so a steering write cannot race native restoration.
//!
//! Each change is logged: `capture worker cam0: exposure 156 -> 187 (luma 82 of the faces, target 110)`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::exposure::{AutoExposure, Exposure, Settings, luma, luma_in, luma_plane};

/// How often the loop offers a frame.
pub const OFFER_NS: u64 = 1_000_000_000;
/// Face boxes older than this are stale: the frame's whole picture is measured instead.
pub const FACES_FRESH: Duration = Duration::from_secs(1);

/// A frame offered to the steering thread.
struct Sample {
    bytes: Vec<u8>,
}

/// The capture loop's end of a camera's auto exposure.
pub struct Steer {
    tx: Option<SyncSender<Sample>>,
    thread: Option<std::thread::JoinHandle<()>>,
    faces_stop: Arc<AtomicBool>,
    faces_task: Option<tokio::task::JoinHandle<()>>,
    last_offer_ns: Option<u64>,
}

impl Steer {
    /// Offers a frame at `now_ns` (monotonic): at most one a second is copied and sent; if the thread is still busy
    /// with the last one, this one is skipped.
    pub fn offer(&mut self, now_ns: u64, bytes: &[u8]) {
        if self.last_offer_ns.is_some_and(|t| now_ns.saturating_sub(t) < OFFER_NS) {
            return;
        }
        self.last_offer_ns = Some(now_ns);
        if let Some(tx) = &self.tx {
            let _ = tx.try_send(Sample { bytes: bytes.to_vec() });
        }
    }
}

impl Drop for Steer {
    fn drop(&mut self) {
        self.faces_stop.store(true, Ordering::Relaxed);
        if let Some(task) = self.faces_task.take() {
            task.abort();
        }
        self.tx.take();
        // Finish every control write before the capture source is torn down.
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Also runs when Tokio aborts a subscription while dropping its runtime.
struct StopOnDrop(Arc<AtomicBool>);
impl Drop for StopOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// What a camera's steering needs from the worker.
pub struct SteerSetup {
    /// The camera's name in the manifest (`cam0`): its log lines, and its faces in `face.boxes`.
    pub camera: String,
    /// Its device node (`/dev/video0`).
    pub node: String,
    /// The payload shape it delivers (`MJPEG`, `UYVY`).
    pub shape: String,
    /// Its frame size and row stride.
    pub width: u32,
    /// Its height.
    pub height: u32,
    /// Bytes per row (raw formats).
    pub stride: u32,
    /// Nanoseconds between frames: the longest exposure that keeps the frame rate.
    pub frame_interval_ns: u64,
    /// How it steers.
    pub settings: Settings,
    /// The broker's socket, for `face.boxes`.
    pub socket: PathBuf,
    /// Manifest watched for live controls and steering preferences.
    pub manifest: PathBuf,
    /// Initial settings, already read by the worker.
    pub video: crate::manifest::VideoSettings,
}

/// The faces last seen on the camera: their boxes and when.
type Faces = Arc<Mutex<Option<(Instant, Vec<[f32; 4]>)>>>;

/// How often a camera's steering says where it stands.
pub const STATUS_NS: u64 = 60_000_000_000;

/// What a camera's steering last saw, for its once-a-minute line, and which failures it has said already.
#[derive(Default)]
struct Status {
    last: Option<(crate::exposure::Luma, &'static str, i64)>,
    last_report_ns: Option<u64>,
    no_decode: bool,
    no_get: bool,
    no_set: bool,
}

/// Says a failure the first time only (`said` marks it).
fn say_once(camera: &str, what: &str, said: &mut bool) {
    if !std::mem::replace(said, true) {
        eprintln!("capture worker {camera}: auto exposure: {what} (said once)");
    }
}

impl Status {
    /// Once a minute: the luma measured last, of what, the exposure, and what the controller did with it.
    fn report(&mut self, setup: &SteerSetup, auto: &AutoExposure, now_ns: u64) {
        if self.last_report_ns.is_some_and(|t| now_ns.saturating_sub(t) < STATUS_NS) {
            return;
        }
        self.last_report_ns = Some(now_ns);
        match self.last {
            Some((l, of, exposure)) => eprintln!(
                "capture worker {}: auto exposure: luma {:.0} of {of} ({:.1}% clipped), target {:.0}, exposure {exposure} \
                 (cap {}): {}",
                setup.camera,
                l.mean,
                l.clipped * 100.0,
                setup.settings.target,
                auto.cap(),
                auto.last()
            ),
            None => eprintln!("capture worker {}: auto exposure: nothing measured yet", setup.camera),
        }
    }
}

/// The startup decision used by the capture worker; OFF must restore native controls without starting steering.
#[cfg(test)]
pub(crate) fn exposure_mode<T>(enabled: bool, start: impl FnOnce() -> Option<T>, restore: impl FnOnce()) -> Option<T> {
    if enabled {
        start()
    } else {
        restore();
        None
    }
}

/// Manual-only edits preserve the current native mode and the running exposure controller.
fn needs_mode_setup(
    first: bool,
    current: &crate::manifest::VideoSettings,
    next: &crate::manifest::VideoSettings,
) -> bool {
    first
        || next.auto_exposure != current.auto_exposure
        || next.hold_fps != current.hold_fps
        || next.exposure_target != current.exposure_target
}

/// Keeps failed mode setup pending independently of successful manual control application.
#[derive(Default)]
struct SetupRetry {
    remaining: u8,
    mode_pending: bool,
}

impl SetupRetry {
    fn request(&mut self, mode: bool) {
        self.remaining = 3;
        self.mode_pending |= mode;
    }

    fn finish(&mut self, mode_ok: bool, controls_ok: bool) {
        if mode_ok {
            self.mode_pending = false;
        }
        self.remaining = if !self.mode_pending && controls_ok { 0 } else { self.remaining.saturating_sub(1) };
    }
}

/// Starts the live-control coordinator, including optional steering. Returns None only if its thread cannot start.
#[must_use]
pub fn start(mut setup: SteerSetup, rt: &tokio::runtime::Handle) -> Option<Steer> {
    let faces: Faces = Arc::new(Mutex::new(None));
    let faces_stop = Arc::new(AtomicBool::new(false));
    let faces_task = follow_faces(&setup, rt, Arc::clone(&faces), Arc::clone(&faces_stop));
    let (tx, rx) = sync_channel::<Sample>(1);
    let name = format!("steer-{}", setup.camera);
    let spawned = std::thread::Builder::new().name(name).spawn(move || {
        let mut current = setup.video.clone();
        let mut device = None;
        let mut auto = None;
        let mut previous = std::collections::BTreeMap::new();
        let mut first = true;
        let mut retry = SetupRetry::default();
        let epoch = Instant::now();
        let mut checked = Instant::now();
        let mut status = Status::default();
        loop {
            if first || checked.elapsed() >= Duration::from_secs(1) {
                checked = Instant::now();
                let next = std::fs::read_to_string(&setup.manifest)
                    .ok()
                    .and_then(|text| crate::manifest::Manifest::parse(&text).ok())
                    .and_then(|manifest| manifest.devices.into_iter().find(|d| d.name == setup.camera && d.enabled))
                    .and_then(|d| d.video())
                    .unwrap_or_else(|| current.clone());
                let reconfigure = needs_mode_setup(first, &current, &next);
                let changed = reconfigure || next.controls != current.controls;
                if changed {
                    retry.request(reconfigure);
                }
                if retry.remaining > 0 {
                    let mut mode_ok = !retry.mode_pending;
                    if retry.mode_pending {
                        // One writer: no old steering write can race native restoration or saved settings.
                        device = None;
                        setup.settings.target = next.exposure_target;
                        if next.auto_exposure {
                            (device, mode_ok) = device::open(&setup);
                        } else {
                            mode_ok = device::restore_native(&setup.camera, &setup.node, next.hold_fps);
                        }
                        auto = device.as_ref().map(|d| {
                            if d.is_exposure() {
                                AutoExposure::new(d.range(), device::UNIT_NS, setup.frame_interval_ns, setup.settings)
                            } else {
                                AutoExposure::uncapped(d.range(), setup.settings)
                            }
                        });
                        status = Status::default();
                    }
                    let controls_ok =
                        device::apply_saved(&setup.camera, &setup.node, &next.controls, &previous, next.auto_exposure);
                    if controls_ok {
                        previous.clone_from(&next.controls);
                    } else {
                        // Track partially applied IDs too, so a subsequent reset can restore their defaults.
                        previous.extend(next.controls.clone());
                    }
                    retry.finish(mode_ok, controls_ok);
                    current = next;
                    first = false;
                }
            }
            let sample = match rx.recv_timeout(Duration::from_millis(200)) {
                Ok(sample) => sample,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            };
            let (Some(device), Some(auto)) = (device.as_mut(), auto.as_mut()) else { continue };
            let knob = device.knob_name();
            let now = u64::try_from(epoch.elapsed().as_nanos()).unwrap_or(u64::MAX);
            status.report(&setup, auto, now);
            let Some((plane, w, h)) = luma_plane(&setup.shape, &sample.bytes, setup.width, setup.height, setup.stride)
            else {
                say_once(&setup.camera, "a frame didn't decode for its luma", &mut status.no_decode);
                continue;
            };
            let boxes = faces
                .lock()
                .ok()
                .and_then(|f| f.as_ref().filter(|(at, _)| at.elapsed() < FACES_FRESH).map(|(_, b)| b.clone()));
            let (measured, of) = match boxes.as_deref().and_then(|b| luma_in(&plane, w, h, b)) {
                Some(l) => (l, "the faces"),
                None => match luma(&plane, 1) {
                    Some(l) => (l, "the frame"),
                    None => continue,
                },
            };
            let Some(current) = device.get() else {
                say_once(&setup.camera, "the exposure can't be read", &mut status.no_get);
                continue;
            };
            status.last = Some((measured, of, current));
            if let Some(next) = auto.step(now, current, measured) {
                // Changed only when the camera reads it back: a write it ignored would otherwise look taken.
                match device.set(next).map(|()| device.get()) {
                    Ok(Some(got)) if got == next => eprintln!(
                        "capture worker {}: {knob} {current} -> {next} (luma {:.0} of {of}, {:.1}% clipped, target \
                         {:.0})",
                        setup.camera,
                        measured.mean,
                        measured.clipped * 100.0,
                        setup.settings.target
                    ),
                    Ok(got) => {
                        auto.refused();
                        let why = format!("the camera kept its {knob} at {got:?} after {next} was written");
                        say_once(&setup.camera, &why, &mut status.no_set);
                    }
                    Err(e) => {
                        auto.refused();
                        say_once(&setup.camera, &format!("the camera refused {knob} {next}: {e}"), &mut status.no_set);
                    }
                }
            }
        }
    });
    match spawned {
        Ok(thread) => Some(Steer {
            tx: Some(tx),
            thread: Some(thread),
            faces_stop,
            faces_task: Some(faces_task),
            last_offer_ns: None,
        }),
        Err(e) => {
            faces_stop.store(true, Ordering::Relaxed);
            faces_task.abort();
            eprintln!("capture worker: no auto exposure, the thread didn't start: {e}");
            None
        }
    }
}

/// Follows the `face.boxes` ring for this camera's face boxes, on a connection of its own; without a face worker (the
/// ring isn't registered) the frame is measured whole.
fn follow_faces(
    setup: &SteerSetup,
    rt: &tokio::runtime::Handle,
    faces: Faces,
    ended: Arc<AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    use continuum_client::{Client, ClientOptions, Start, SubscribeSpec};

    let (socket, camera) = (setup.socket.clone(), setup.camera.clone());
    let guard = StopOnDrop(Arc::clone(&ended));
    rt.spawn(async move {
        let _guard = guard;
        let Ok(client) = Client::connect(&socket, ClientOptions::new(format!("capture-{camera}-exposure"))).await
        else {
            return;
        };
        let mut sub = match client.subscribe(&SubscribeSpec::new(["face.boxes"], Start::Live)).await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("capture worker {camera}: exposure measures the whole frame (no face.boxes: {e})");
                return;
            }
        };
        let Some(mut reader) = sub.ring_reader(&client, None, continuum_shm::ReadFrom::Latest) else {
            eprintln!("capture worker {camera}: exposure measures the whole frame (face.boxes has no ring)");
            return;
        };
        // The ring is followed on a blocking thread through its sequence word. The subscription carries no events (a
        // ring stream sends no ingests); it ends with the connection, a broker restart included, and so does this.
        let stop = Arc::clone(&ended);
        let follow = tokio::task::spawn_blocking(move || {
            let mut buf = vec![0u8; reader.stride()];
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                reader.wait(Duration::from_millis(100));
                while !stop.load(Ordering::Relaxed) {
                    let Ok(Some(_)) = reader.read(&mut buf) else { break };
                    if let Some(boxes) = continuum_client::length_prefixed(&buf).and_then(|r| face_boxes(r, &camera))
                        && let Ok(mut f) = faces.lock()
                    {
                        *f = Some((Instant::now(), boxes));
                    }
                }
            }
        });
        while sub.next().await.is_some() {}
        ended.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = follow.await;
        drop(client);
    })
}

/// The face boxes of a `face.boxes` record for `camera`, or None for another camera's or a record that isn't one.
#[must_use]
pub fn face_boxes(record: &[u8], camera: &str) -> Option<Vec<[f32; 4]>> {
    let v: serde_json::Value = serde_json::from_slice(record).ok()?;
    if v.get("cam")?.as_str()? != camera {
        return None;
    }
    let boxes = v
        .get("faces")?
        .as_array()?
        .iter()
        .filter_map(|f| {
            let b = f.get("bbox")?.as_array()?;
            #[allow(clippy::cast_possible_truncation)]
            let n = |i: usize| b.get(i).and_then(serde_json::Value::as_f64).map(|x| x as f32);
            Some([n(0)?, n(1)?, n(2)?, n(3)?])
        })
        .collect();
    Some(boxes)
}

#[cfg(target_os = "linux")]
mod device {
    use v4l::control::{Control, Value};

    use super::SteerSetup;
    use crate::exposure::{Exposure, Range};
    use crate::video::{CID_EXPOSURE_AUTO, EXPOSURE_SHUTTER_PRIORITY};

    /// `V4L2_CID_EXPOSURE_ABSOLUTE`, in 100 µs units.
    pub const CID_EXPOSURE_ABSOLUTE: u32 = 0x009a_0902;
    /// `V4L2_CID_BRIGHTNESS`.
    pub const CID_BRIGHTNESS: u32 = 0x0098_0900;
    /// The absolute exposure control's unit.
    pub const UNIT_NS: u64 = 100_000;

    /// What steering turns: the exposure where the driver lets it be set, else the camera's brightness.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Knob {
        /// `exposure_time_absolute`, capped at the frame period.
        Exposure,
        /// `brightness`: the camera's own processing level, on top of the gain its auto mode sets.
        Brightness,
    }

    /// A V4L2 camera's steered control, through a handle of its own on the node (controls can be set while another
    /// handle streams).
    pub struct Device {
        device: v4l::Device,
        range: Range,
        id: u32,
        /// Which control it is.
        pub knob: Knob,
    }

    /// The camera's steerable control, or None (said why) when it has none.
    ///
    /// In shutter priority (which capture sets for `hold_fps`) the exposure is ours and the gain the camera's, where
    /// the driver allows it. Linux's UVC driver lets `exposure_time_absolute` be set only in Manual mode, so a camera
    /// that offers just Auto and Shutter Priority (the Elgato Facecam) has it inactive (a write is refused, EACCES):
    /// then its brightness is steered instead, the camera's gain doing the rest as before.
    ///
    /// A camera with neither (a kernel that permanently disabled the Facecam's brightness after one stalled
    /// query) is put in its own auto mode instead (`fall_back`), said once at WARN, so its picture still adapts.
    ///
    /// The bool says whether the mode setup is settled: false only for a failure worth retrying.
    pub fn open(setup: &SteerSetup) -> (Option<Device>, bool) {
        let mut device = match v4l::Device::with_path(&setup.node) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("capture worker {}: no auto exposure: {e}", setup.camera);
                return (None, false);
            }
        };
        let (knob, control) = match steering_control(&mut device) {
            Ok(Some(control)) => control,
            Ok(None) => {
                let (settled, report) = fall_back(&mut device);
                eprintln!("capture worker {}: WARN {report}", setup.camera);
                return (None, settled);
            }
            Err(error) => {
                eprintln!("capture worker {}: no auto exposure: {error}", setup.camera);
                return (None, false);
            }
        };
        let range =
            Range { min: control.minimum, max: control.maximum, step: i64::try_from(control.step).unwrap_or(1) };
        let what = match knob {
            Knob::Exposure => format!("exposure {} to {} (x100 us)", range.min, range.max),
            Knob::Brightness => {
                format!("brightness {} to {} (its exposure can't be set in this mode)", range.min, range.max)
            }
        };
        eprintln!("capture worker {}: auto exposure toward luma {:.0}, by {what}", setup.camera, setup.settings.target);
        (Some(Device { device, range, id: control.id, knob }), true)
    }

    /// A camera steering can't turn: its own automatic exposure mode instead (frame rate not held, so it may drop in
    /// low light), and the report to log, naming the controls the driver disabled and how to get them back. Settled
    /// unless the camera refused a mode it advertises (worth retrying).
    fn fall_back(controls: &mut impl NativeControls) -> (bool, String) {
        let descriptions = match controls.descriptions() {
            Ok(descriptions) => descriptions,
            Err(error) => return (false, format!("no auto exposure, and cannot inspect controls: {error}")),
        };
        let disabled: Vec<&str> = descriptions
            .iter()
            .filter(|c| c.flags.contains(v4l::control::Flags::DISABLED) && c.typ != v4l::control::Type::CtrlClass)
            .map(|c| c.name.as_str())
            .collect();
        let why = if disabled.is_empty() {
            "neither exposure nor brightness can be set".to_owned()
        } else {
            format!(
                "neither exposure nor brightness can be set; the driver disabled {} (replug the camera or reload \
                 uvcvideo to probe them again)",
                disabled.join(", ")
            )
        };
        let (settled, mode) = match native_mode(controls, &descriptions, false) {
            Ok(mode) => (true, format!("{mode} instead (its frame rate may drop in low light)")),
            Err(error) => (!error.starts_with("camera refused"), error),
        };
        (settled, format!("no auto exposure steering: {why}; {mode}"))
    }

    /// Unsupported shutter priority keeps the native mode and steers brightness instead.
    /// Supported mode switches must verify; unsupported modes retain their advertised writable controls.
    /// None when the camera has no writable exposure or brightness.
    fn steering_control(
        controls: &mut impl NativeControls,
    ) -> Result<Option<(Knob, v4l::control::Description)>, String> {
        let descriptions = controls.descriptions()?;
        let priority = descriptions.iter().any(|c| {
            c.id == CID_EXPOSURE_AUTO
                && c.typ == v4l::control::Type::Menu
                && c.items
                    .as_ref()
                    .is_some_and(|items| items.iter().any(|(index, _)| i64::from(*index) == EXPOSURE_SHUTTER_PRIORITY))
        });
        if priority {
            set_verified(controls, CID_EXPOSURE_AUTO, EXPOSURE_SHUTTER_PRIORITY)?;
        }
        let mut descriptions = controls.descriptions()?;
        descriptions.sort_by_key(|c| c.id != CID_EXPOSURE_ABSOLUTE);
        Ok(descriptions
            .into_iter()
            .find(|c| {
                c.typ == v4l::control::Type::Integer
                    && writable(c)
                    && (c.id == CID_EXPOSURE_ABSOLUTE || c.id == CID_BRIGHTNESS)
            })
            .map(|control| {
                (if control.id == CID_EXPOSURE_ABSOLUTE { Knob::Exposure } else { Knob::Brightness }, control)
            }))
    }

    /// The same V4L2 adapter is driven with synthetic descriptions/reads/writes in the tests below.
    trait NativeControls {
        fn descriptions(&mut self) -> Result<Vec<v4l::control::Description>, String>;
        fn read(&mut self, id: u32) -> Result<i64, String>;
        fn write(&mut self, id: u32, value: i64) -> Result<(), String>;
    }

    impl NativeControls for v4l::Device {
        fn descriptions(&mut self) -> Result<Vec<v4l::control::Description>, String> {
            self.query_controls().map_err(|error| error.to_string())
        }
        fn read(&mut self, id: u32) -> Result<i64, String> {
            scalar(&self.control(id).map_err(|error| error.to_string())?.value)
        }
        fn write(&mut self, id: u32, value: i64) -> Result<(), String> {
            // An integer write sets a boolean control too: V4L2 takes the same value field for both.
            self.set_control(Control { id, value: Value::Integer(value) }).map_err(|error| error.to_string())
        }
    }

    /// A control's value as the integer saved settings use. The v4l crate reads a boolean control (auto white
    /// balance, say) as `Boolean`: that's 0 or 1 here, or a saved boolean would never verify.
    fn scalar(value: &Value) -> Result<i64, String> {
        match value {
            Value::Integer(value) => Ok(*value),
            Value::Boolean(value) => Ok(i64::from(*value)),
            _ => Err("camera returned a non-scalar control value".into()),
        }
    }

    fn writable(control: &v4l::control::Description) -> bool {
        use v4l::control::Flags;
        !control
            .flags
            .intersects(Flags::DISABLED | Flags::READ_ONLY | Flags::INACTIVE | Flags::GRABBED | Flags::WRITE_ONLY)
    }

    fn legal(control: &v4l::control::Description, value: i64) -> bool {
        control.step > 0
            && value >= control.minimum
            && value <= control.maximum
            && (i128::from(value) - i128::from(control.minimum)) % i128::from(control.step) == 0
    }

    fn set_verified(controls: &mut impl NativeControls, id: u32, value: i64) -> Result<(), String> {
        if controls.read(id).ok() == Some(value) {
            return Ok(());
        }
        controls.write(id, value)?;
        if controls.read(id)? == value {
            Ok(())
        } else {
            Err("camera did not retain the requested control value".into())
        }
    }

    /// AUTO and APERTURE_PRIORITY automate exposure time; SHUTTER_PRIORITY leaves exposure time manual.
    /// https://docs.kernel.org/userspace-api/media/v4l/ext-ctrls-camera.html
    fn native_mode(
        controls: &mut impl NativeControls,
        descriptions: &[v4l::control::Description],
        hold_fps: bool,
    ) -> Result<&'static str, String> {
        use v4l::control::Type;
        let mode = descriptions
            .iter()
            .find(|control| {
                control.id == CID_EXPOSURE_AUTO && control.typ == Type::Menu && control.step == 1 && writable(control)
            })
            .ok_or_else(|| "no readable/writable exposure-auto menu advertised; mode left unchanged".to_owned())?;
        let advertised = |value: i64| {
            legal(mode, value)
                && mode.items.as_ref().is_some_and(|items| items.iter().any(|(index, _)| i64::from(*index) == value))
        };
        let mut candidates = Vec::new();
        if matches!(mode.default, 0 | 3) && advertised(mode.default) {
            candidates.push(mode.default);
        }
        for value in [0, 3] {
            if advertised(value) && !candidates.contains(&value) {
                candidates.push(value);
            }
        }
        let mut why = "neither AUTO nor APERTURE_PRIORITY advertised; mode left unchanged".to_owned();
        if hold_fps && advertised(EXPOSURE_SHUTTER_PRIORITY) {
            match set_verified(controls, CID_EXPOSURE_AUTO, EXPOSURE_SHUTTER_PRIORITY) {
                Ok(()) => return Ok("SHUTTER_PRIORITY exposure held for hold_fps"),
                Err(error) => why = format!("camera refused SHUTTER_PRIORITY for hold_fps: {error}"),
            }
        }
        for value in candidates {
            match set_verified(controls, CID_EXPOSURE_AUTO, value) {
                Ok(()) => {
                    return Ok(if value == 0 {
                        "native AUTO exposure restored"
                    } else {
                        "native APERTURE_PRIORITY exposure restored"
                    });
                }
                Err(error) => why = format!("camera refused native auto mode {value}: {error}"),
            }
        }
        Err(why)
    }

    fn restore_controls_status(controls: &mut impl NativeControls, hold_fps: bool) -> (bool, String) {
        let mut success = true;
        let descriptions = match controls.descriptions() {
            Ok(descriptions) => descriptions,
            Err(error) => return (false, format!("cannot inspect native controls: {error}; controls left unchanged")),
        };
        let mode = match native_mode(controls, &descriptions, hold_fps) {
            Ok(mode) => mode.to_owned(),
            Err(error) => {
                success = !descriptions.iter().any(|c| c.id == CID_EXPOSURE_AUTO && writable(c));
                error
            }
        };
        // Mode changes can change control flags. Never use stale brightness metadata or invent a midpoint.
        let brightness = match controls.descriptions() {
            Ok(descriptions) => {
                match descriptions.iter().find(|control| {
                    control.id == CID_BRIGHTNESS
                        && control.typ == v4l::control::Type::Integer
                        && writable(control)
                        && legal(control, control.default)
                }) {
                    Some(control) => match set_verified(controls, CID_BRIGHTNESS, control.default) {
                        Ok(()) => format!("native brightness default {} restored", control.default),
                        Err(error) => {
                            success = false;
                            format!("camera refused native brightness default: {error}")
                        }
                    },
                    None => "no writable valid native brightness default advertised; brightness left unchanged".into(),
                }
            }
            Err(error) => {
                success = false;
                format!("cannot inspect native brightness default: {error}; brightness left unchanged")
            }
        };
        (success, format!("{mode}; {brightness}"))
    }

    #[cfg(test)]
    fn restore_controls(controls: &mut impl NativeControls, hold_fps: bool) -> String {
        restore_controls_status(controls, hold_fps).1
    }

    pub(super) fn restore_native(camera: &str, node: &str, hold_fps: bool) -> bool {
        match v4l::Device::with_path(node) {
            Ok(mut device) => {
                let (success, report) = restore_controls_status(&mut device, hold_fps);
                eprintln!("capture worker {camera}: exposure steering OFF: {report}");
                success
            }
            Err(error) => {
                eprintln!("capture worker {camera}: exposure steering OFF: cannot restore native controls: {error}");
                false
            }
        }
    }

    /// Applies saved scalar controls through a control-only handle, without opening a stream or changing its format.
    pub(super) fn apply_saved(
        camera: &str,
        node: &str,
        saved: &std::collections::BTreeMap<String, i64>,
        previous: &std::collections::BTreeMap<String, i64>,
        steering: bool,
    ) -> bool {
        if saved.is_empty() && previous.is_empty() {
            return true;
        }
        let Ok(mut device) = v4l::Device::with_path(node) else {
            eprintln!("capture worker {camera}: saved camera controls: cannot open {node}");
            return false;
        };
        let errors = apply_controls(&mut device, saved, previous, steering);
        let applied = errors.is_empty();
        for error in errors {
            eprintln!("capture worker {camera}: saved camera control: {error}");
        }
        applied
    }

    fn apply_controls(
        controls: &mut impl NativeControls,
        saved: &std::collections::BTreeMap<String, i64>,
        previous: &std::collections::BTreeMap<String, i64>,
        steering: bool,
    ) -> Vec<String> {
        use v4l::control::Type;
        let mut ids: Vec<_> = saved.keys().chain(previous.keys()).collect();
        ids.sort();
        ids.dedup();
        // Mode switches unlock dependent controls. Query flags again before every value.
        ids.sort_by_key(|key| {
            let id = key.parse::<u32>().unwrap_or(0);
            (!matches!(id, CID_EXPOSURE_AUTO | 0x0098_090c | 0x009a_090c | 0x0098_0912), id)
        });
        let mut errors = Vec::new();
        for key in ids {
            let result = (|| {
                let id = key.parse::<u32>().map_err(|_| format!("invalid decimal ID {key}"))?;
                if steering && matches!(id, CID_EXPOSURE_AUTO | CID_EXPOSURE_ABSOLUTE | CID_BRIGHTNESS) {
                    return Ok(());
                }
                let descriptions = controls.descriptions()?;
                let c = descriptions.iter().find(|c| c.id == id).ok_or_else(|| format!("{id}: not advertised"))?;
                // A removed override can become inactive after restoring its automatic mode.
                // That mode now owns its value; do not retain an impossible reset forever.
                if !saved.contains_key(key) && c.flags.contains(v4l::control::Flags::INACTIVE) {
                    return Ok(());
                }
                let value = saved.get(key).copied().unwrap_or(c.default);
                let scalar = matches!(c.typ, Type::Integer | Type::Boolean | Type::Menu | Type::IntegerMenu);
                let menu = !matches!(c.typ, Type::Menu | Type::IntegerMenu)
                    || c.items.as_ref().is_some_and(|items| items.iter().any(|(index, _)| i64::from(*index) == value));
                if !scalar || !writable(c) || !legal(c, value) || !menu {
                    return Err(format!("{id}: value {value} is unavailable in the current camera mode"));
                }
                set_verified(controls, id, value).map_err(|error| format!("{id}: {error}"))
            })();
            if let Err(error) = result {
                errors.push(error);
            }
        }
        errors
    }

    impl Device {
        /// Whether it steers the exposure (capped at the frame period), not the brightness.
        #[must_use]
        pub fn is_exposure(&self) -> bool {
            self.knob == Knob::Exposure
        }

        /// The control's name, for the log.
        #[must_use]
        pub fn knob_name(&self) -> &'static str {
            match self.knob {
                Knob::Exposure => "exposure",
                Knob::Brightness => "brightness",
            }
        }
    }

    impl Exposure for Device {
        fn range(&self) -> Range {
            self.range
        }

        fn get(&mut self) -> Option<i64> {
            match self.device.control(self.id) {
                Ok(Control { value: Value::Integer(v), .. }) => Some(v),
                _ => None,
            }
        }

        fn set(&mut self, value: i64) -> Result<(), String> {
            self.device
                .set_control(Control { id: self.id, value: Value::Integer(value) })
                .map_err(|e| format!("{e} (errno {})", e.raw_os_error().unwrap_or(0)))
        }
    }
    #[cfg(test)]
    mod native_tests {
        use super::*;
        use std::collections::BTreeSet;
        use v4l::control::{Description, Flags, MenuItem, Type};

        struct FakeV4l {
            mode: i64,
            brightness: i64,
            menu: Vec<i64>,
            mode_default: i64,
            mode_step: u64,
            mode_flags: u32,
            brightness_flags: u32,
            brightness_after_native: bool,
            brightness_range: (i64, i64, u64, i64),
            refused: BTreeSet<(u32, i64)>,
            ignored: bool,
            query_failed: bool,
            writes: Vec<(u32, i64)>,
        }

        impl Default for FakeV4l {
            fn default() -> Self {
                Self {
                    mode: EXPOSURE_SHUTTER_PRIORITY,
                    brightness: 35,
                    menu: vec![0, 1, 2, 3],
                    mode_default: 0,
                    mode_step: 1,
                    mode_flags: 0,
                    brightness_flags: 0,
                    brightness_after_native: false,
                    brightness_range: (0, 255, 1, 128),
                    refused: BTreeSet::new(),
                    ignored: false,
                    query_failed: false,
                    writes: vec![],
                }
            }
        }

        impl NativeControls for FakeV4l {
            fn descriptions(&mut self) -> Result<Vec<Description>, String> {
                if self.query_failed {
                    return Err("synthetic query failure".into());
                }
                let (minimum, maximum, step, default) = self.brightness_range;
                Ok(vec![
                    Description {
                        id: CID_EXPOSURE_AUTO,
                        typ: Type::Menu,
                        name: "synthetic exposure menu".into(),
                        minimum: 0,
                        maximum: 3,
                        step: self.mode_step,
                        default: self.mode_default,
                        flags: Flags::from(self.mode_flags),
                        items: Some(
                            self.menu
                                .iter()
                                .map(|value| (u32::try_from(*value).unwrap(), MenuItem::Name(format!("mode {value}"))))
                                .collect(),
                        ),
                    },
                    Description {
                        id: CID_BRIGHTNESS,
                        typ: Type::Integer,
                        name: "synthetic brightness".into(),
                        minimum,
                        maximum,
                        step,
                        default,
                        flags: Flags::from(if self.brightness_after_native && self.mode == EXPOSURE_SHUTTER_PRIORITY {
                            Flags::INACTIVE.bits()
                        } else {
                            self.brightness_flags
                        }),
                        items: None,
                    },
                ])
            }
            fn read(&mut self, id: u32) -> Result<i64, String> {
                Ok(if id == CID_EXPOSURE_AUTO { self.mode } else { self.brightness })
            }
            fn write(&mut self, id: u32, value: i64) -> Result<(), String> {
                self.writes.push((id, value));
                if self.refused.contains(&(id, value)) {
                    return Err("synthetic refusal".into());
                }
                if !self.ignored {
                    if id == CID_EXPOSURE_AUTO {
                        self.mode = value;
                    } else {
                        self.brightness = value;
                    }
                }
                Ok(())
            }
        }

        #[test]
        fn off_restores_native_auto_and_the_advertised_brightness_without_starting_steering() {
            let mut camera = FakeV4l::default();
            let steer: Option<()> = super::super::exposure_mode(
                false,
                || panic!("OFF must never start steering"),
                || {
                    let report = restore_controls(&mut camera, false);
                    assert!(report.contains("native AUTO exposure restored"));
                },
            );
            assert!(steer.is_none());
            assert_eq!((camera.mode, camera.brightness), (0, 128));
            assert_eq!(camera.writes, [(CID_EXPOSURE_AUTO, 0), (CID_BRIGHTNESS, 128)]);
            camera.writes.clear();
            restore_controls(&mut camera, false);
            assert!(camera.writes.is_empty(), "already-native defaults need no redundant writes");
        }

        #[test]
        fn with_hold_fps_off_holds_shutter_priority_instead_of_native_auto() {
            // cam0's Facecam: only AUTO and SHUTTER_PRIORITY, left in AUTO when the hold at open didn't take.
            let mut facecam = FakeV4l { menu: vec![0, 2], mode: 0, ..Default::default() };
            let report = restore_controls(&mut facecam, true);
            assert!(report.contains("SHUTTER_PRIORITY exposure held for hold_fps"), "{report}");
            assert_eq!(facecam.mode, EXPOSURE_SHUTTER_PRIORITY);
            assert_eq!(facecam.writes, [(CID_EXPOSURE_AUTO, EXPOSURE_SHUTTER_PRIORITY), (CID_BRIGHTNESS, 128)]);
        }

        #[test]
        fn a_hold_the_camera_refuses_or_lacks_still_restores_native_auto() {
            let refused = [(CID_EXPOSURE_AUTO, EXPOSURE_SHUTTER_PRIORITY)].into_iter().collect();
            let mut refusing = FakeV4l { menu: vec![0, 2], mode: 0, refused, ..Default::default() };
            assert!(restore_controls(&mut refusing, true).contains("native AUTO exposure restored"));
            assert_eq!(refusing.mode, 0);
            let mut lacking = FakeV4l { menu: vec![0, 1, 3], mode: 1, ..Default::default() };
            assert!(restore_controls(&mut lacking, true).contains("native AUTO exposure restored"));
            assert_eq!(lacking.mode, 0);
        }

        #[test]
        fn restoration_uses_only_advertised_automatic_modes_and_never_a_manual_default() {
            for (menu, default, expected) in [(vec![0, 1, 2, 3], 3, 3), (vec![0, 1, 2], 1, 0), (vec![1, 2, 3], 1, 3)] {
                let mut camera = FakeV4l { menu, mode_default: default, ..Default::default() };
                restore_controls(&mut camera, false);
                assert_eq!(camera.mode, expected);
                assert_eq!(camera.writes[0], (CID_EXPOSURE_AUTO, expected));
            }
            let mut unsupported = FakeV4l { menu: vec![1, 2], mode_default: 2, ..Default::default() };
            assert!(
                restore_controls(&mut unsupported, false).contains("neither AUTO nor APERTURE_PRIORITY advertised")
            );
            assert_eq!(unsupported.mode, 2);
            assert_eq!(unsupported.writes, [(CID_BRIGHTNESS, 128)], "only the known brightness default may change");
        }

        #[test]
        fn refused_and_ignored_controls_report_partial_restore_without_guessing_values() {
            let mut camera = FakeV4l { refused: [(CID_EXPOSURE_AUTO, 0)].into_iter().collect(), ..Default::default() };
            assert!(restore_controls(&mut camera, false).contains("native APERTURE_PRIORITY exposure restored"));
            assert_eq!(camera.mode, 3);
            let mut ignored = FakeV4l { ignored: true, ..Default::default() };
            let report = restore_controls(&mut ignored, false);
            assert!(report.contains("did not retain"));
            assert!(!report.contains("exposure restored"));
            assert_eq!((ignored.mode, ignored.brightness), (2, 35));
        }

        #[test]
        fn unavailable_controls_or_invalid_defaults_are_not_written() {
            for flag in [Flags::DISABLED, Flags::READ_ONLY, Flags::INACTIVE, Flags::GRABBED, Flags::WRITE_ONLY] {
                let mut camera =
                    FakeV4l { mode_flags: flag.bits(), brightness_flags: flag.bits(), ..Default::default() };
                let report = restore_controls(&mut camera, false);
                assert!(report.contains("left unchanged"));
                assert!(camera.writes.is_empty());
            }
            for default in [-1, 256, 127] {
                let mut camera = FakeV4l { brightness_range: (0, 255, 2, default), ..Default::default() };
                assert!(restore_controls(&mut camera, false).contains("brightness left unchanged"));
                assert!(!camera.writes.iter().any(|(id, _)| *id == CID_BRIGHTNESS));
            }
            let mut failed = FakeV4l { query_failed: true, ..Default::default() };
            assert!(restore_controls(&mut failed, false).contains("cannot inspect native controls"));
            assert!(failed.writes.is_empty());
        }

        #[test]
        fn on_starts_steering_without_resetting_native_defaults() {
            assert_eq!(
                super::super::exposure_mode(true, || Some(7), || panic!("ON must not restore controls")),
                Some(7)
            );
        }
        #[test]
        fn invalid_metadata_zero_step_brightness_is_not_written() {
            let mut camera = FakeV4l { brightness_range: (0, 255, 0, 128), ..Default::default() };
            let report = restore_controls(&mut camera, false);
            assert!(
                !camera.writes.iter().any(|(id, _)| *id == CID_BRIGHTNESS),
                "invalid zero-step metadata must not invent step one"
            );
            assert_eq!(camera.brightness, 35);
            assert!(report.contains("brightness left unchanged"));
        }

        #[test]
        fn invalid_metadata_menu_steps_do_not_select_an_exposure_mode() {
            for mode_step in [0, 2] {
                let mut camera = FakeV4l { mode_step, ..Default::default() };
                restore_controls(&mut camera, false);
                assert!(
                    !camera.writes.iter().any(|(id, _)| *id == CID_EXPOSURE_AUTO),
                    "invalid menu step {mode_step} must not select a guessed mode"
                );
                assert_eq!(camera.mode, EXPOSURE_SHUTTER_PRIORITY);
            }
        }

        #[test]
        fn unsupported_shutter_priority_keeps_native_mode_and_steers_brightness() {
            let mut camera = FakeV4l { mode: 0, menu: vec![0, 1], ..Default::default() };
            let (knob, control) = steering_control(&mut camera).unwrap().unwrap();
            assert_eq!((knob, control.id), (Knob::Brightness, CID_BRIGHTNESS));
            assert_eq!(camera.mode, 0);
            assert!(camera.writes.is_empty());
        }

        #[test]
        fn a_boolean_control_reads_as_the_zero_or_one_a_saved_setting_holds() {
            assert_eq!(scalar(&Value::Boolean(false)), Ok(0));
            assert_eq!(scalar(&Value::Boolean(true)), Ok(1));
            assert_eq!(scalar(&Value::Integer(-7)), Ok(-7));
            assert!(scalar(&Value::String("x".into())).is_err());
        }

        /// cam0's Facecam after a kernel disabled its brightness: Auto and Shutter Priority only, exposure
        /// time not settable in Shutter Priority, brightness DISABLED.
        fn knobless_facecam() -> FakeV4l {
            FakeV4l { menu: vec![0, 2], brightness_flags: Flags::DISABLED.bits(), ..Default::default() }
        }

        #[test]
        fn a_camera_with_no_knob_falls_back_to_its_own_auto_mode_and_names_what_was_disabled() {
            let mut facecam = knobless_facecam();
            assert!(steering_control(&mut facecam).unwrap().is_none());
            facecam.writes.clear();
            let (settled, report) = fall_back(&mut facecam);
            assert!(settled, "{report}");
            assert_eq!(facecam.mode, 0, "the camera's own Auto Mode, not a fixed Shutter Priority exposure");
            assert_eq!(facecam.writes, [(CID_EXPOSURE_AUTO, 0)], "brightness is disabled: never written");
            assert!(report.contains("synthetic brightness") && report.contains("replug"), "{report}");
        }

        #[test]
        fn a_knobless_camera_without_disabled_controls_gets_no_replug_advice() {
            let mut camera =
                FakeV4l { menu: vec![0, 2], brightness_flags: Flags::INACTIVE.bits(), ..Default::default() };
            let (settled, report) = fall_back(&mut camera);
            assert!(settled && camera.mode == 0, "{report}");
            assert!(!report.contains("replug"), "{report}");
        }

        #[test]
        fn a_refused_fallback_is_retried_and_one_with_no_auto_mode_is_not() {
            let refused = [(CID_EXPOSURE_AUTO, 0)].into_iter().collect();
            let mut refusing = FakeV4l { refused, ..knobless_facecam() };
            assert!(!fall_back(&mut refusing).0);
            assert_eq!(refusing.mode, EXPOSURE_SHUTTER_PRIORITY);
            let mut manual_only = FakeV4l { menu: vec![1, 2], ..knobless_facecam() };
            let (settled, report) = fall_back(&mut manual_only);
            assert!(settled, "nothing to retry: {report}");
            assert_eq!(manual_only.mode, EXPOSURE_SHUTTER_PRIORITY);
        }

        #[test]
        fn unsupported_shutter_priority_preserves_writable_absolute_exposure_fallback() {
            struct Absolute(FakeV4l);
            impl NativeControls for Absolute {
                fn descriptions(&mut self) -> Result<Vec<Description>, String> {
                    let mut descriptions = self.0.descriptions()?;
                    descriptions.push(Description {
                        id: CID_EXPOSURE_ABSOLUTE,
                        typ: Type::Integer,
                        name: "synthetic exposure".into(),
                        minimum: 1,
                        maximum: 100,
                        step: 1,
                        default: 20,
                        flags: Flags::empty(),
                        items: None,
                    });
                    Ok(descriptions)
                }
                fn read(&mut self, id: u32) -> Result<i64, String> {
                    self.0.read(id)
                }
                fn write(&mut self, id: u32, value: i64) -> Result<(), String> {
                    self.0.write(id, value)
                }
            }
            let mut camera = Absolute(FakeV4l { mode: 1, menu: vec![0, 1], ..Default::default() });
            assert_eq!(steering_control(&mut camera).unwrap().unwrap().0, Knob::Exposure);
            assert_eq!(camera.0.mode, 1);
            assert!(camera.0.writes.is_empty());
        }

        #[test]
        fn advertised_shutter_priority_must_read_back_before_steering_starts() {
            let mut camera = FakeV4l { mode: 0, ignored: true, ..Default::default() };
            assert!(steering_control(&mut camera).is_err());
            camera.ignored = false;
            assert_eq!(steering_control(&mut camera).unwrap().unwrap().0, Knob::Brightness);
            assert_eq!(camera.mode, EXPOSURE_SHUTTER_PRIORITY);
        }

        #[test]
        fn removed_inactive_override_settles_but_present_inactive_value_is_rejected() {
            let mut camera = FakeV4l { brightness_after_native: true, ..Default::default() };
            let previous = [(CID_BRIGHTNESS.to_string(), 47)].into();
            assert!(apply_controls(&mut camera, &Default::default(), &previous, false).is_empty());
            assert_eq!(apply_controls(&mut camera, &previous, &Default::default(), false).len(), 1);
        }

        #[test]
        fn native_setup_failure_is_reported_and_can_recover() {
            let mut camera = FakeV4l { query_failed: true, ..Default::default() };
            assert!(!restore_controls_status(&mut camera, false).0);
            camera.query_failed = false;
            assert!(restore_controls_status(&mut camera, false).0);
            camera.mode = EXPOSURE_SHUTTER_PRIORITY;
            camera.ignored = true;
            assert!(!restore_controls_status(&mut camera, false).0);
        }

        #[test]
        fn saved_controls_switch_mode_before_requerying_dependent_flags() {
            let mut camera = FakeV4l { brightness_after_native: true, ..Default::default() };
            let saved = [(CID_BRIGHTNESS.to_string(), 47), (CID_EXPOSURE_AUTO.to_string(), 0)].into();
            assert!(apply_controls(&mut camera, &saved, &Default::default(), false).is_empty());
            assert_eq!((camera.mode, camera.brightness), (0, 47));
            assert_eq!(camera.writes[0], (CID_EXPOSURE_AUTO, 0));
        }

        #[test]
        fn custom_steering_protects_its_mode_and_brightness_from_saved_values() {
            let mut camera = FakeV4l::default();
            let saved = [(CID_BRIGHTNESS.to_string(), 47), (CID_EXPOSURE_AUTO.to_string(), 0)].into();
            assert!(apply_controls(&mut camera, &saved, &Default::default(), true).is_empty());
            assert!(camera.writes.is_empty());
        }

        #[test]
        fn removed_override_restores_advertised_default_and_refuses_illegal_values() {
            let mut camera = FakeV4l { brightness: 47, ..Default::default() };
            let previous = [(CID_BRIGHTNESS.to_string(), 47)].into();
            assert!(apply_controls(&mut camera, &Default::default(), &previous, false).is_empty());
            assert_eq!(camera.brightness, 128);
            let saved = [(CID_BRIGHTNESS.to_string(), 256), (CID_EXPOSURE_AUTO.to_string(), 4)].into();
            assert_eq!(apply_controls(&mut camera, &saved, &Default::default(), false).len(), 2);
            assert_eq!(camera.brightness, 128);
        }

        #[test]
        fn native_mode_requeries_brightness_flags_and_preserves_a_signed_driver_default() {
            let mut camera =
                FakeV4l { brightness_after_native: true, brightness_range: (-64, 64, 4, -12), ..Default::default() };
            let report = restore_controls(&mut camera, false);
            assert_eq!((camera.mode, camera.brightness), (0, -12));
            assert!(report.contains("native brightness default -12 restored"));
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod device {
    use super::SteerSetup;
    use crate::exposure::{Exposure, Range};

    /// The exposure unit (unused off Linux).
    pub const UNIT_NS: u64 = 100_000;

    /// No device control here yet (macOS: AVFoundation's custom exposure).
    pub struct Device;

    impl Device {
        pub fn is_exposure(&self) -> bool {
            true
        }
        pub fn knob_name(&self) -> &'static str {
            "exposure"
        }
    }

    pub(super) fn restore_native(_camera: &str, _node: &str, _hold_fps: bool) -> bool {
        // No application steering runs here, so the native exposure mode is already untouched.
        true
    }

    pub(super) fn apply_saved(
        camera: &str,
        _node: &str,
        saved: &std::collections::BTreeMap<String, i64>,
        _previous: &std::collections::BTreeMap<String, i64>,
        _steering: bool,
    ) -> bool {
        if !saved.is_empty() {
            eprintln!("capture worker {camera}: saved camera controls require V4L2 on Linux");
        }
        saved.is_empty()
    }

    pub fn open(setup: &SteerSetup) -> (Option<Device>, bool) {
        eprintln!("capture worker {}: auto exposure isn't built for this OS yet: the camera's own runs", setup.camera);
        (None, true)
    }

    impl Exposure for Device {
        fn range(&self) -> Range {
            Range { min: 0, max: 0, step: 1 }
        }
        fn get(&mut self) -> Option<i64> {
            None
        }
        fn set(&mut self, _value: i64) -> Result<(), String> {
            Err("not built for this OS".to_owned())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coordinator_drop_joins_controls_and_cancels_silent_follower_before_runtime_drop() {
        let (done, completion) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(1).enable_all().build().unwrap();
            let stop = Arc::new(AtomicBool::new(false));
            let stopped = Arc::clone(&stop);
            let (started, ready) = std::sync::mpsc::channel();
            let guard = StopOnDrop(Arc::clone(&stop));
            let task = rt.spawn(async move {
                let _guard = guard;
                tokio::task::spawn_blocking(move || {
                    started.send(()).unwrap();
                    while !stopped.load(Ordering::Relaxed) {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                })
                .await
                .unwrap();
            });
            ready.recv_timeout(Duration::from_secs(2)).unwrap();
            let (tx, rx) = sync_channel::<Sample>(1);
            let finished = Arc::new(AtomicBool::new(false));
            let finished_thread = Arc::clone(&finished);
            let thread = std::thread::spawn(move || {
                while rx.recv().is_ok() {}
                finished_thread.store(true, Ordering::Relaxed);
            });
            let steer = Steer {
                tx: Some(tx),
                thread: Some(thread),
                faces_stop: stop,
                faces_task: Some(task),
                last_offer_ns: None,
            };
            drop(steer);
            assert!(finished.load(Ordering::Relaxed));
            drop(rt);
            done.send(()).unwrap();
        });
        completion.recv_timeout(Duration::from_secs(4)).expect("coordinator/runtime shutdown must finish");
    }

    #[test]
    fn runtime_abort_signals_a_silent_blocking_follower() {
        let (done, completion) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(1).enable_all().build().unwrap();
            let stop = Arc::new(AtomicBool::new(false));
            let stopped = Arc::clone(&stop);
            let (started, ready) = std::sync::mpsc::channel();
            let guard = StopOnDrop(Arc::clone(&stop));
            rt.spawn(async move {
                let _guard = guard;
                tokio::task::spawn_blocking(move || {
                    started.send(()).unwrap();
                    while !stopped.load(Ordering::Relaxed) {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                })
                .await
                .unwrap();
            });
            ready.recv_timeout(Duration::from_secs(2)).unwrap();
            drop(rt);
            assert!(stop.load(Ordering::Relaxed));
            done.send(()).unwrap();
        });
        completion.recv_timeout(Duration::from_secs(4)).expect("runtime abort must release its blocking follower");
    }

    #[test]
    fn transient_mode_failure_retries_even_when_empty_manual_map_succeeds() {
        let mut retry = SetupRetry::default();
        retry.request(true);
        retry.finish(false, true);
        assert!(retry.mode_pending);
        assert_eq!(retry.remaining, 2);
        retry.finish(true, true);
        assert!(!retry.mode_pending);
        assert_eq!(retry.remaining, 0);
    }

    #[test]
    fn permanent_mode_failure_is_bounded_and_manual_edits_do_not_reconfigure_completed_mode() {
        let mut retry = SetupRetry::default();
        retry.request(true);
        let mut calls = 0;
        while retry.remaining > 0 {
            calls += 1;
            retry.finish(false, true);
        }
        assert_eq!(calls, 3);
        assert!(retry.mode_pending);
        retry.request(true);
        retry.finish(true, true);
        retry.request(false);
        assert!(!retry.mode_pending);
        retry.finish(true, false);
        assert!(!retry.mode_pending);
        assert_eq!(retry.remaining, 2);
    }

    #[test]
    fn live_off_manual_on_transitions_only_reconfigure_exposure_when_needed() {
        let mut off = crate::manifest::VideoSettings { auto_exposure: false, ..Default::default() };
        assert!(needs_mode_setup(true, &off, &off));
        let mut manual = off.clone();
        manual.controls.insert("9963776".into(), 47);
        assert!(!needs_mode_setup(false, &off, &manual));
        off = manual.clone();
        manual.auto_exposure = true;
        assert!(needs_mode_setup(false, &off, &manual));
        assert!(needs_mode_setup(false, &manual, &off));
    }

    #[test]
    fn exposure_off_restores_native_controls_without_starting_steering() {
        use crate::video::Controls;
        struct FakeV4l {
            exposure: i64,
            brightness: i64,
        }
        impl Controls for FakeV4l {
            fn set(&mut self, id: u32, value: i64) -> bool {
                if id == crate::video::CID_EXPOSURE_AUTO {
                    self.exposure = value;
                } else {
                    self.brightness = value;
                }
                true
            }
        }
        let mut camera = FakeV4l { exposure: crate::video::EXPOSURE_SHUTTER_PRIORITY, brightness: 35 };
        let started = std::cell::Cell::new(false);
        let steer = exposure_mode(
            false,
            || {
                started.set(true);
                Some(())
            },
            || {
                camera.set(crate::video::CID_EXPOSURE_AUTO, 0);
                camera.set(0x0098_0900, 128);
            },
        );
        assert!(steer.is_none());
        assert!(!started.get(), "OFF must not start a steering thread or face subscription");
        assert_eq!(
            (camera.exposure, camera.brightness),
            (0, 128),
            "OFF must restore the camera's own auto mode and brightness"
        );
    }

    #[test]
    fn a_face_boxes_record_gives_its_cameras_boxes_and_nothing_for_another() {
        let rec = br#"{"cam":"cam0","faces":[{"bbox":[0.1,0.2,0.3,0.4],"name":"Dana"},{"bbox":[0.5,0.5,0.6]}]}"#;
        assert_eq!(face_boxes(rec, "cam0"), Some(vec![[0.1, 0.2, 0.3, 0.4]]), "a box with three numbers is skipped");
        assert_eq!(face_boxes(rec, "cam1"), None);
        assert_eq!(face_boxes(b"not json", "cam0"), None);
    }

    #[test]
    fn the_loop_offers_at_most_a_frame_a_second_and_never_waits() {
        let (tx, rx) = sync_channel(1);
        let mut s = Steer {
            tx: Some(tx),
            thread: None,
            faces_stop: Arc::new(AtomicBool::new(false)),
            faces_task: None,
            last_offer_ns: None,
        };
        s.offer(0, &[1]);
        s.offer(500_000_000, &[2]); // half a second later: not offered
        s.offer(1_000_000_000, &[3]); // the channel is full: dropped, not waited on
        assert_eq!(rx.try_recv().map(|x| x.bytes).ok(), Some(vec![1]));
        assert!(rx.try_recv().is_err());
    }
}
