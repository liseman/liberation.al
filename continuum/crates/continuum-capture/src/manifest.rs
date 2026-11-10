//! The device manifest: which devices to capture, how to recognize each, and its settings (docs/CONTINUUM.md,
//! "One process per physical device").
//!
//! ```toml
//! [supervisor]
//! settle_ms = 1500
//!
//! [audio]
//! exclude = ["arducam", "iphone", "ipad", "continuity"]
//!
//! # Every camera's settings, which a camera's own `video = {...}` overrides key by key.
//! [video]
//! width = 1280
//! height = 720
//! fps = 15
//!
//! [[device]]
//! name = "facecam"
//! kind = "camera"
//! match = { name = "Elgato Facecam" }
//! video = { width = 1920, height = 1080, fps = 30 }
//!
//! [[device]]
//! name = "wave3"
//! kind = "microphone"
//! match = { name = "Elgato Wave 3" }
//! ```

use std::collections::BTreeSet;

use serde::Deserialize;

/// The whole manifest.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// The supervisor's timings.
    #[serde(default)]
    pub supervisor: SupervisorSettings,
    /// Rules for every microphone.
    #[serde(default)]
    pub audio: AudioRules,
    /// Every camera's settings: each camera's own `video` table overrides these key by key (merged at parse, so
    /// [`Device::video`] already has the result).
    #[serde(default)]
    pub video: Option<VideoSettings>,
    /// The devices, in order.
    #[serde(default, rename = "device")]
    pub devices: Vec<Device>,
    /// The rigs that carry cameras together (#5): a camera's `mount.rig` names one. Capture doesn't use them; the
    /// spatial readers do (spatial/rig.py).
    #[serde(default, rename = "rig")]
    pub rigs: Vec<Rig>,
}

/// Cameras that move together, such as a backpack's. Its pose in a place comes from its pose track, not from here.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Rig {
    /// Unique among rigs: letters, digits, `-`, and `_`.
    pub name: String,
}

/// How a camera is mounted (#5), read by spatial/rig.py; capture doesn't use it. A camera on a rig gives where it
/// sits on the rig and which way it looks; a fixed camera gives its height and whether it looks outdoors. Which of
/// the fields go together is spatial/rig.py's to check.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Mount {
    /// The rig it's on, by its `[[rig]]` name; none for a fixed camera.
    pub rig: Option<String>,
    /// Where it sits in the rig's frame, metres: x right, y down, z forward.
    pub at: Option<[f64; 3]>,
    /// Which way it looks on the rig, degrees: yaw to the right, pitch up, roll clockwise.
    pub turn: Option<[f64; 3]>,
    /// A rig camera's horizontal field of view, degrees.
    pub hfov_deg: Option<f64>,
    /// A fixed camera's height from the floor to its lens, metres.
    pub height_m: Option<f64>,
    /// A fixed camera that looks outdoors.
    #[serde(default)]
    pub outdoor: bool,
}

/// How the supervisor paces opens and restarts.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields, default)]
pub struct SupervisorSettings {
    /// After a worker opens its device (or fails to), how long before the next worker may open one: concurrent UVC
    /// opens fail, and so do opens right after another.
    pub settle_ms: u64,
    /// A worker's first restart waits this long; each restart after doubles it, up to `backoff_max_ms`.
    pub backoff_initial_ms: u64,
    /// The longest a restart waits.
    pub backoff_max_ms: u64,
    /// A worker that recorded this long (from opening its device) before exiting is restarted after
    /// `backoff_initial_ms` again. 15 s: a network camera on a flapping link recorded 30 to 60 s at a time, each exit
    /// an RTSP timing check, and at 60 s every run counted as quick, so each restart waited the full 60 s cap and
    /// driveway and patio recorded 10 to 35 percent of 11-07's afternoon and evening. A run this long opened
    /// its device and recorded, so restarting it again soon costs nothing; a failed open or an instant death still
    /// doubles the wait.
    pub healthy_after_ms: u64,
    /// A worker that hasn't said its device is open this long after starting is stopped and counted as failed.
    pub open_timeout_ms: u64,
    /// A worker asked to stop that hasn't exited this long after is killed; one that outlives the kill by as long
    /// again is given up on. Long enough for macOS to stop a UVC session and an audio unit: a kill mid-teardown
    /// can wedge the device until it is power-cycled.
    pub stop_grace_ms: u64,
    /// macOS only: the aggregate raw camera estimate allowed before workers start. Unknown USB topology pools all
    /// configured cameras as one bus, even for requested MJPEG. Native modes cost at least two bytes/pixel,
    /// or four for RGB/unknown formats. The conservative 40 MB/s default leaves room below USB 2's signaling limit;
    /// it is a policy estimate, not measured bus speed.
    /// Raise explicitly only after verifying the cameras' transport and a faster independent bus; never set zero.
    pub mac_raw_budget_bytes_per_second: u64,
}

impl Default for SupervisorSettings {
    fn default() -> Self {
        Self {
            settle_ms: 1_500,
            backoff_initial_ms: 2_000,
            backoff_max_ms: 60_000,
            healthy_after_ms: 15_000,
            open_timeout_ms: 15_000,
            stop_grace_ms: 10_000,
            mac_raw_budget_bytes_per_second: crate::mac_bandwidth::DEFAULT_BUDGET_BYTES_PER_SECOND,
        }
    }
}

/// Which microphones to use, by name, for every microphone entry.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields, default)]
pub struct AudioRules {
    /// Case-insensitive pieces of a device name to skip (a camera's own microphone, which wedges its USB hub; a
    /// phone's continuity microphone).
    pub exclude: Vec<String>,
    /// On Linux: take PipeWire's sources when PipeWire is running, else raw ALSA devices, never both (one microphone
    /// opened through two aliases wedged PipeWire in v1).
    pub prefer_pipewire: bool,
}

impl Default for AudioRules {
    fn default() -> Self {
        Self { exclude: ["arducam", "iphone", "ipad", "continuity"].map(String::from).to_vec(), prefer_pipewire: true }
    }
}

impl AudioRules {
    /// Whether a microphone named `name` is skipped.
    #[must_use]
    pub fn excludes(&self, name: &str) -> bool {
        let name = name.to_lowercase();
        self.exclude.iter().any(|piece| name.contains(&piece.to_lowercase()))
    }
}

/// What kind of device an entry is, which decides the worker that runs it.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum DeviceKind {
    /// A UVC or AVFoundation camera (with its microphone, if it has one and it isn't excluded).
    Camera,
    /// A configured H.264 RTSP or RTSPS camera, independent of USB hotplug.
    Rtsp,
    /// A microphone.
    Microphone,
    /// A ReSpeaker XVF3800 array: audio, plus direction of arrival through its control table.
    Xvf3800,
    /// A Teensy on USB serial: its sensors' samples.
    Teensy,
}

impl DeviceKind {
    /// The streams a device of this kind registers, by suffix (`facecam.video`): what `residual_offset_ns` may name.
    #[must_use]
    pub fn streams(self) -> &'static [&'static str] {
        match self {
            Self::Camera => &["video", "audio"],
            Self::Rtsp => &["video"],
            Self::Microphone => &["audio"],
            Self::Xvf3800 => &["audio", "doa"],
            Self::Teensy => &["imu", "env", "light"],
        }
    }
}

/// How to recognize a device when it appears. Every field given must match.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DeviceMatch {
    /// A case-insensitive piece of the device's name (the V4L2 card name, the audio device's description).
    pub name: Option<String>,
    /// The USB vendor and product, like `0fd9:0078`.
    pub usb: Option<String>,
    /// A stable path or id, like a `/dev/v4l/by-id/...` link or a PipeWire node name.
    pub path: Option<String>,
}

/// A camera's capture settings.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct VideoSettings {
    /// Width asked for first.
    pub width: u32,
    /// Height asked for first.
    pub height: u32,
    /// Frames per second asked for.
    pub fps: u32,
    /// The device's native payload: `MJPEG` (the default, for every camera) or `UYVY`.
    pub format: String,
    /// Sizes to fall back to, in order, when the camera won't open at the one asked for.
    pub fallback: Vec<(u32, u32)>,
    /// Keep the frame rate in dim light: the camera raises gain instead of lengthening exposure (V4L2 only).
    pub hold_fps: bool,
    /// The camera is mounted upside down: readers flip it (it's a flag in `CaptureInfo`, not a pixel operation).
    pub flip: bool,
    /// Video horizon used to provision broker slots: ten seconds compressed, two by default for native raw.
    pub ring_seconds: f32,
    /// No frame for this long means the camera stalled: the worker exits and the supervisor restarts it.
    pub stall_ms: u64,
    /// Steer exposure toward `exposure_target`; false restores advertised native auto/default controls.
    pub auto_exposure: bool,
    /// The mean luma (0 to 255) auto exposure steers toward: the faces' when some are in view, else the frame's.
    pub exposure_target: f32,
    /// Persisted V4L2 scalar control values, keyed by decimal control ID.
    pub controls: std::collections::BTreeMap<String, i64>,
    /// MJPEG-to-H.264 capture encoding. A local MJPEG camera encodes by default ([`Self::encoding_for`]);
    /// `enabled = false` keeps its MJPEG.
    pub encoding: Option<VideoEncodingSettings>,
}

impl VideoSettings {
    /// The encoding a device of `kind` uses: its own settings when set, else on with the defaults for a local
    /// MJPEG camera, so MJPEG never reaches Tier 1. A build without the `ffmpeg` feature can't encode, so
    /// there the default is MJPEG as captured.
    #[must_use]
    pub fn encoding_for(&self, kind: DeviceKind) -> Option<VideoEncodingSettings> {
        self.encoding.clone().or_else(|| {
            (cfg!(feature = "ffmpeg") && kind == DeviceKind::Camera && self.format == "MJPEG")
                .then(VideoEncodingSettings::default)
        })
    }
}

impl Default for VideoSettings {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            fps: 30,
            format: "MJPEG".into(),
            fallback: vec![(960, 540), (640, 480)],
            hold_fps: true,
            flip: false,
            ring_seconds: 10.0,
            stall_ms: 15_000,
            auto_exposure: true,
            exposure_target: 110.0,
            controls: Default::default(),
            encoding: None,
        }
    }
}

/// The explicitly selected capture encoder. Native capture is selected by leaving encoding disabled.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VideoEncodingBackend {
    /// NVIDIA hardware H.264 encoding, requiring an available local NVENC device.
    H264Nvenc,
}

/// Encoding of a camera's MJPEG frames into its canonical H.264 video stream: on unless `enabled = false`.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields, default)]
pub struct VideoEncodingSettings {
    /// False keeps the camera's MJPEG as captured.
    pub enabled: bool,
    /// Preferred hardware encoder; capture falls back loudly to software H.264.
    pub backend: VideoEncodingBackend,
    /// CUDA device ordinal, not a machine-specific device identifier.
    pub gpu: u32,
    /// NVENC constant quality, from 0 through 51; 36 is a candidate requiring validation.
    pub cq: u32,
    /// Maximum requested interval between keyframes, milliseconds.
    pub keyframe_interval_ms: u32,
}

impl Default for VideoEncodingSettings {
    fn default() -> Self {
        Self { enabled: true, backend: VideoEncodingBackend::H264Nvenc, gpu: 0, cq: 36, keyframe_interval_ms: 2000 }
    }
}

impl VideoEncodingSettings {
    /// Validate encoding settings for their effective device kind and native capture format.
    ///
    /// # Errors
    /// Refuses unsupported inputs and out-of-range values.
    pub fn check(&self, kind: DeviceKind, format: &str) -> Result<(), &'static str> {
        if kind != DeviceKind::Camera || format != "MJPEG" {
            return Err("video.encoding requires a camera with MJPEG capture");
        }
        if i32::try_from(self.gpu).is_err() || self.cq > 51 || !(100..=10_000).contains(&self.keyframe_interval_ms) {
            return Err("video.encoding requires gpu 0..2147483647, cq 0..51, and keyframe_interval_ms 100..10000");
        }
        Ok(())
    }
}

/// RTSP transport, with TCP interleaving preferred for loss-resistant local camera streams.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RtspTransport {
    /// RTP interleaved in the RTSP TCP connection (also used by RTSPS).
    #[default]
    Tcp,
    /// RTP over UDP, where the camera requires it.
    Udp,
}

/// Network camera settings. Debug deliberately excludes the complete URL, including opaque path tokens.
#[derive(Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields, default)]
pub struct RtspSettings {
    /// Private local URL, potentially containing a password or opaque path token.
    pub url: String,
    /// RTP transport.
    pub transport: RtspTransport,
    /// Maximum time to open and probe the stream.
    pub connect_timeout_ms: u64,
    /// Maximum time without a network packet before reconnecting.
    pub read_timeout_ms: u64,
    /// Maximum encoded access unit accepted into one Tier 0 slot.
    pub max_access_unit_bytes: u32,
    /// Hard Tier 0 payload and slot-metadata budget; the time horizon is best effort within it.
    pub ring_bytes: u64,
    /// Source dimensions are never scaled; larger inputs are refused.
    pub max_width: u32,
    /// Source dimensions are never scaled; larger inputs are refused.
    pub max_height: u32,
}

impl Default for RtspSettings {
    fn default() -> Self {
        Self {
            url: String::new(),
            transport: RtspTransport::Tcp,
            connect_timeout_ms: 10_000,
            read_timeout_ms: 10_000,
            max_access_unit_bytes: 4 * 1024 * 1024,
            ring_bytes: 64 * 1024 * 1024,
            max_width: 4096,
            max_height: 4096,
        }
    }
}

impl std::fmt::Debug for RtspSettings {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("RtspSettings")
            .field("url", &"<redacted>")
            .field("transport", &self.transport)
            .field("connect_timeout_ms", &self.connect_timeout_ms)
            .field("read_timeout_ms", &self.read_timeout_ms)
            .field("max_access_unit_bytes", &self.max_access_unit_bytes)
            .field("ring_bytes", &self.ring_bytes)
            .field("max_width", &self.max_width)
            .field("max_height", &self.max_height)
            .finish()
    }
}

impl RtspSettings {
    fn check(&self) -> Result<(), &'static str> {
        let authority = self
            .url
            .strip_prefix("rtsp://")
            .or_else(|| self.url.strip_prefix("rtsps://"))
            .and_then(|tail| tail.split('/').next())
            .and_then(|host| host.rsplit('@').next());
        if self.url.len() > 4096 || self.url.chars().any(char::is_control) || authority.is_none_or(str::is_empty) {
            return Err("rtsp.url must be a bounded RTSP or RTSPS URL (value redacted)");
        }
        if !(100..=60_000).contains(&self.connect_timeout_ms) || !(100..=60_000).contains(&self.read_timeout_ms) {
            return Err("RTSP connect/read timeouts must be between 100 and 60000 milliseconds");
        }
        if !(64 * 1024..=16 * 1024 * 1024).contains(&self.max_access_unit_bytes) {
            return Err("RTSP max_access_unit_bytes must be between 64 KiB and 16 MiB");
        }
        if self.ring_bytes < 2 * (u64::from(self.max_access_unit_bytes).next_power_of_two() + 64)
            || self.ring_bytes > 1024 * 1024 * 1024
        {
            return Err("RTSP ring_bytes must hold two maximum access units and be at most 1 GiB");
        }
        if self.max_width == 0 || self.max_width > 8192 || self.max_height == 0 || self.max_height > 8192 {
            return Err("RTSP maximum dimensions must be between 1 and 8192");
        }
        Ok(())
    }
}

/// A microphone's capture settings.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct AudioSettings {
    /// Each capture block, milliseconds: what a live reader waits for (10 to 20).
    pub block_ms: u32,
    /// Seconds of audio the stream's ring holds.
    pub ring_seconds: f32,
    /// Seconds per `IngestNotice`: the catalog's granularity for audio, not the live path's.
    pub batch_seconds: f32,
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self { block_ms: 10, ring_seconds: 10.0, batch_seconds: 1.0 }
    }
}

/// One device to capture.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Device {
    /// Unique: its streams are named after it (`facecam.video`, `wave3.audio`).
    pub name: String,
    /// Retain the private configuration and stream identity while deliberately excluding capture.
    #[serde(default = "enabled_by_default")]
    pub enabled: bool,
    /// Paused for privacy from the dashboard's header: the supervisor keeps no worker on it, so the device
    /// is not opened and nothing from it is captured, until `paused` is false again. Followed live, about once a
    /// second, without restarting other devices.
    #[serde(default)]
    pub paused: bool,
    /// What it is.
    pub kind: DeviceKind,
    /// How to recognize it.
    #[serde(default, rename = "match")]
    pub matches: DeviceMatch,
    /// H.264 network input. The URL is private and never included in diagnostics.
    #[serde(default)]
    pub rtsp: Option<RtspSettings>,
    /// For a camera.
    #[serde(default)]
    pub video: Option<VideoSettings>,
    /// For a microphone, or a camera's or an array's microphone.
    #[serde(default)]
    pub audio: Option<AudioSettings>,
    /// Each stream's fixed latency, measured once per rig by a clap-and-flash check, nanoseconds to subtract from its
    /// stamps, by stream suffix (`video`, `audio`).
    #[serde(default)]
    pub residual_offset_ns: std::collections::BTreeMap<String, i64>,
    /// A fisheye camera's lens calibration. Capture doesn't use it; the Python readers do (see-actions
    /// rectifies each person through it, see/fisheye.py, and spatial/cameras.py leaves fisheye cameras out of
    /// fixed-camera places). It's declared here because a device's unknown keys are refused, so that a mistyped
    /// setting (`pasued = true`) stops capture rather than being ignored.
    #[serde(default)]
    pub fisheye: Option<FisheyeCalibration>,
    /// A wide (not fisheye) lens's radial distortion, which spatial/lens.py undoes before MASt3R, the
    /// footprints, and the splats use the camera's frames. Capture doesn't use it.
    #[serde(default)]
    pub lens: Option<LensCalibration>,
    /// Overlays the camera burns into its frames (a timestamp, its name, a logo), as rectangles `[x1, y1, x2, y2]` in
    /// fractions of the frame: spatial/lens.py paints them over and leaves them out. Capture doesn't use them.
    #[serde(default)]
    pub masks: Vec<[f64; 4]>,
    /// How the camera is mounted: on a rig, or fixed (#5). Capture doesn't use it.
    #[serde(default)]
    pub mount: Option<Mount>,
}

/// A wide lens's calibration, as `tools/calibrate_lens.py` writes it:
/// `lens = { width = 896, height = 512, k = [448.0, 448.0, 447.5, 255.5], division = -0.31 }`, a one-parameter division
/// model (spatial/lens.py).
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LensCalibration {
    /// The frame size, in pixels, the calibration was fitted on.
    pub width: u32,
    /// Its height.
    pub height: u32,
    /// The focal lengths and principal point, in pixels: fx, fy, cx, cy.
    pub k: [f64; 4],
    /// The division model's lambda: negative for a barrel.
    pub division: f64,
}

/// A fisheye lens's calibration, as `tools/calibrate_fisheye.py` writes it:
/// `fisheye = { width = 1920, height = 1080, k = [fx, fy, cx, cy], d = [k1, k2, k3, k4], rms = 0.4 }`.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FisheyeCalibration {
    /// The frame size, in pixels, the calibration was fitted on.
    pub width: u32,
    /// Its height.
    pub height: u32,
    /// The focal lengths and principal point, in pixels: fx, fy, cx, cy.
    pub k: [f64; 4],
    /// OpenCV's four fisheye distortion terms.
    pub d: [f64; 4],
    /// The fit's RMS reprojection error, in pixels.
    #[serde(default)]
    pub rms: Option<f64>,
}

fn enabled_by_default() -> bool {
    true
}

impl Device {
    /// Its video settings, or the defaults for a camera.
    #[must_use]
    pub fn video(&self) -> Option<VideoSettings> {
        self.video
            .clone()
            .or_else(|| matches!(self.kind, DeviceKind::Camera | DeviceKind::Rtsp).then(VideoSettings::default))
    }

    /// Its audio settings, or the defaults for a device that captures audio.
    #[must_use]
    pub fn audio(&self) -> Option<AudioSettings> {
        self.audio
            .clone()
            .or_else(|| matches!(self.kind, DeviceKind::Microphone | DeviceKind::Xvf3800).then(AudioSettings::default))
    }

    /// The stream a suffix names, like `facecam.video`.
    #[must_use]
    pub fn stream_name(&self, suffix: &str) -> String {
        format!("{}.{suffix}", self.name)
    }
}

/// Why a manifest can't be used.
#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    /// It isn't valid TOML of this shape.
    #[error("the device manifest doesn't parse: {0}")]
    Parse(String),
    /// Two devices share a name.
    #[error("two devices are named {0:?}")]
    DuplicateName(String),
    /// A device can't be recognized: it has no match field.
    #[error("device {0:?} has nothing to match it by (give match.name, match.usb, or match.path)")]
    NoMatch(String),
    /// A setting is out of range.
    #[error("device {0:?}: {1}")]
    Invalid(String, String),
}

impl From<toml::de::Error> for ManifestError {
    fn from(error: toml::de::Error) -> Self {
        // TOML diagnostics otherwise include source excerpts and invalid values, both of which may be URLs.
        let category =
            if error.message().starts_with("unknown field") { "unknown field" } else { "invalid TOML syntax or value" };
        let location = error.span().map_or(String::new(), |span| format!(" near byte {}", span.start));
        Self::Parse(format!("{category}{location}; source values omitted"))
    }
}

impl Manifest {
    /// Parses and checks a manifest.
    ///
    /// # Errors
    /// [`ManifestError`] when it doesn't parse or doesn't make sense.
    pub fn parse(text: &str) -> Result<Self, ManifestError> {
        Self::parse_native(text, cfg!(target_os = "macos"))
    }

    fn parse_native(text: &str, native_raw: bool) -> Result<Self, ManifestError> {
        let mut table: toml::Table = toml::from_str(text)?;
        merge_video_defaults(&mut table);
        apply_video_ring_defaults(&mut table, native_raw);
        let mut m: Self = table.try_into()?;
        m.check()?;
        for device in &mut m.devices {
            if device.kind == DeviceKind::Rtsp {
                device.matches.path = Some(format!("rtsp:{}", device.name));
            }
        }
        Ok(m)
    }

    fn check(&self) -> Result<(), ManifestError> {
        let s = &self.supervisor;
        let bad = |why: &str| Err(ManifestError::Invalid("[supervisor]".into(), why.into()));
        if s.mac_raw_budget_bytes_per_second == 0 {
            return bad("mac_raw_budget_bytes_per_second must be positive (macOS auto-fit budget)");
        }
        if s.backoff_initial_ms < 100 || s.backoff_max_ms < s.backoff_initial_ms {
            return bad("backoff_initial_ms is at least 100, and backoff_max_ms at least that");
        }
        if s.open_timeout_ms < 1_000 || s.stop_grace_ms < 100 {
            return bad("open_timeout_ms is at least 1000, and stop_grace_ms at least 100");
        }
        if let Some(video) = &self.video
            && let Some(encoding) = &video.encoding
        {
            encoding
                .check(DeviceKind::Camera, &video.format)
                .map_err(|why| ManifestError::Invalid("[video]".into(), why.into()))?;
        }
        let mut rigs = BTreeSet::new();
        for r in &self.rigs {
            if r.name.is_empty() || !r.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
                return Err(ManifestError::Invalid(r.name.clone(), "a rig's name is letters, digits, - and _".into()));
            }
            if !rigs.insert(r.name.as_str()) {
                return Err(ManifestError::Invalid(r.name.clone(), "a rig is named twice".into()));
            }
        }
        let mut names = BTreeSet::new();
        for d in &self.devices {
            if let Some(rig) = d.mount.as_ref().and_then(|m| m.rig.as_deref())
                && !rigs.contains(rig)
            {
                return Err(ManifestError::Invalid(d.name.clone(), format!("mount.rig {rig:?} is no [[rig]]")));
            }
            if d.name.is_empty() || !d.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
                return Err(ManifestError::Invalid(d.name.clone(), "a name is letters, digits, - and _".into()));
            }
            if !names.insert(d.name.as_str()) {
                return Err(ManifestError::DuplicateName(d.name.clone()));
            }
            let m = &d.matches;
            if d.kind == DeviceKind::Rtsp {
                if m.name.is_some() || m.usb.is_some() || m.path.is_some() {
                    return Err(ManifestError::Invalid(
                        d.name.clone(),
                        "RTSP devices use their manifest name, not match rules".into(),
                    ));
                }
                let settings = d
                    .rtsp
                    .as_ref()
                    .ok_or_else(|| ManifestError::Invalid(d.name.clone(), "RTSP settings are required".into()))?;
                settings.check().map_err(|message| ManifestError::Invalid(d.name.clone(), message.into()))?;
                if d.audio.is_some() {
                    return Err(ManifestError::Invalid(d.name.clone(), "RTSP capture is video only".into()));
                }
            } else if d.rtsp.is_some() {
                return Err(ManifestError::Invalid(d.name.clone(), "rtsp settings require kind='rtsp'".into()));
            }
            if d.kind != DeviceKind::Rtsp && m.name.is_none() && m.usb.is_none() && m.path.is_none() {
                return Err(ManifestError::NoMatch(d.name.clone()));
            }
            if let Some(usb) = &m.usb
                && parse_usb(usb).is_none()
            {
                return Err(ManifestError::Invalid(d.name.clone(), format!("match.usb {usb:?} isn't vendor:product")));
            }
            if let Some(v) = d.video() {
                if let Some(encoding) = &v.encoding {
                    encoding
                        .check(d.kind, &v.format)
                        .map_err(|why| ManifestError::Invalid(d.name.clone(), why.into()))?;
                }
                if v.width == 0 || v.height == 0 || v.fps == 0 {
                    return Err(ManifestError::Invalid(d.name.clone(), "video needs a width, height, and fps".into()));
                }
                if d.kind != DeviceKind::Rtsp && !matches!(v.format.as_str(), "MJPEG" | "UYVY") {
                    return Err(ManifestError::Invalid(d.name.clone(), format!("video format {:?}", v.format)));
                }
                if v.ring_seconds <= 0.0 {
                    return Err(ManifestError::Invalid(d.name.clone(), "ring_seconds must be positive".into()));
                }
            }
            if let Some(stream) = d.residual_offset_ns.keys().find(|k| !d.kind.streams().contains(&k.as_str())) {
                return Err(ManifestError::Invalid(
                    d.name.clone(),
                    format!("residual_offset_ns names {stream:?}, not one of its streams {:?}", d.kind.streams()),
                ));
            }
            if let Some(a) = d.audio()
                && (a.block_ms == 0
                    || a.ring_seconds <= 0.0
                    || a.batch_seconds <= 0.0
                    || a.batch_seconds > a.ring_seconds)
            {
                return Err(ManifestError::Invalid(
                    d.name.clone(),
                    "audio needs a block, a ring, and a batch no longer than the ring".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Puts the manifest's top-level `[video]` under each camera's own `video` table, the camera's keys winning, so a
/// camera with no `video` of its own gets the manifest's.
fn merge_video_defaults(table: &mut toml::Table) {
    let Some(toml::Value::Table(defaults)) = table.get("video").cloned() else { return };
    let Some(toml::Value::Array(devices)) = table.get_mut("device") else { return };
    for device in devices.iter_mut().filter_map(toml::Value::as_table_mut) {
        if device.get("kind").and_then(toml::Value::as_str) != Some("camera") {
            continue;
        }
        let mut merged = defaults.clone();
        if let Some(toml::Value::Table(own)) = device.get("video") {
            let mut encoding = merged.get("encoding").and_then(toml::Value::as_table).cloned();
            if let (Some(defaults), Some(overrides)) =
                (&mut encoding, own.get("encoding").and_then(toml::Value::as_table))
            {
                defaults.extend(overrides.iter().map(|(k, v)| (k.clone(), v.clone())));
            }
            merged.extend(own.iter().map(|(k, v)| (k.clone(), v.clone())));
            if own.get("encoding").is_none_or(toml::Value::is_table)
                && let Some(encoding) = encoding
            {
                merged.insert("encoding".into(), toml::Value::Table(encoding));
            }
        }
        device.insert("video".into(), toml::Value::Table(merged));
    }
}

/// Raw frames need a shorter default horizon. Apply this after inheritance so explicit global and device values
/// always win, including an explicit ten-second raw ring. AVFoundation delivers raw even when MJPEG was requested.
fn apply_video_ring_defaults(table: &mut toml::Table, native_raw: bool) {
    fn apply(video: &mut toml::Table, native_raw: bool) {
        if !video.contains_key("ring_seconds")
            && (native_raw || video.get("format").and_then(toml::Value::as_str) == Some("UYVY"))
        {
            video.insert("ring_seconds".into(), toml::Value::Float(2.0));
        }
    }
    if let Some(toml::Value::Array(devices)) = table.get_mut("device") {
        for device in devices.iter_mut().filter_map(toml::Value::as_table_mut) {
            if device.get("kind").and_then(toml::Value::as_str) != Some("camera") {
                continue;
            }
            let raw = native_raw
                || device
                    .get("video")
                    .and_then(toml::Value::as_table)
                    .and_then(|v| v.get("format"))
                    .and_then(toml::Value::as_str)
                    == Some("UYVY");
            if raw {
                let video = device.entry("video").or_insert_with(|| toml::Value::Table(toml::Table::new()));
                if let Some(video) = video.as_table_mut() {
                    apply(video, native_raw);
                }
            }
        }
    }
    // Devices already inherited the original table, so a default inserted here cannot override a device's format.
    if let Some(video) = table.get_mut("video").and_then(toml::Value::as_table_mut) {
        apply(video, native_raw);
    }
}

/// `vvvv:pppp` as (vendor, product).
#[must_use]
pub fn parse_usb(text: &str) -> Option<(u16, u16)> {
    let (v, p) = text.split_once(':')?;
    Some((u16::from_str_radix(v, 16).ok()?, u16::from_str_radix(p, 16).ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoding_manifest(video: &str, kind: &str) -> String {
        format!("[[device]]\nname='synthetic'\nkind='{kind}'\nmatch={{name='Synthetic'}}\nvideo={{{video}}}\n")
    }

    #[test]
    fn a_local_mjpeg_camera_encodes_unless_it_opts_out() {
        let unset = Manifest::parse(&encoding_manifest("fps=15", "camera")).unwrap();
        let unset = &unset.devices[0];
        let encodes = cfg!(feature = "ffmpeg");
        assert_eq!(
            unset.video().unwrap().encoding_for(unset.kind).is_some_and(|e| e.enabled),
            encodes,
            "MJPEG never reaches Tier 1 from a build that can encode"
        );
        let table = Manifest::parse(&encoding_manifest("encoding={}", "camera")).unwrap();
        assert!(table.devices[0].video().unwrap().encoding.unwrap().enabled, "an empty table is on");
        let off = Manifest::parse(&encoding_manifest("encoding={enabled=false}", "camera")).unwrap();
        let off = &off.devices[0];
        assert!(!off.video().unwrap().encoding_for(off.kind).unwrap().enabled, "opting out keeps MJPEG");
        let raw = Manifest::parse(&encoding_manifest("format='UYVY'", "camera")).unwrap();
        let raw = &raw.devices[0];
        assert!(raw.video().unwrap().encoding_for(raw.kind).is_none(), "raw cameras are encoded by the tier worker");
    }

    #[test]
    fn camera_controls_keep_decimal_ids_and_signed_values_per_device() {
        let parsed = Manifest::parse(
            "[[device]]\nname='synthetic'\nkind='camera'\nmatch={name='Synthetic'}\nvideo={controls={'9963776'=-12}}\n",
        )
        .unwrap();
        assert_eq!(parsed.devices[0].video().unwrap().controls.get("9963776"), Some(&-12));
        assert!(VideoSettings::default().controls.is_empty());
    }

    #[test]
    fn encoding_inherits_each_key_and_explicit_disable_wins() {
        let text = "[video.encoding]\nenabled=true\ngpu=4\ncq=28\nkeyframe_interval_ms=1500\n\n[[device]]\nname='synthetic'\nkind='camera'\nmatch={name='Synthetic'}\nvideo={fps=15,encoding={enabled=false,cq=36}}\n";
        let parsed = Manifest::parse(text).unwrap();
        let video = parsed.devices[0].video().unwrap();
        let encoding = video.encoding.unwrap();
        assert!(!encoding.enabled);
        assert_eq!((encoding.gpu, encoding.cq, encoding.keyframe_interval_ms), (4, 36, 1500));
        assert_eq!(video.fps, 15);
    }

    #[test]
    fn encoding_rejects_invalid_fields_ranges_and_inputs() {
        for field in [
            "backend='unknown'",
            "gup=1",
            "gpu=2147483648",
            "gpu=-1",
            "cq=52",
            "cq=-1",
            "keyframe_interval_ms=99",
            "keyframe_interval_ms=10001",
            "enabled='yes'",
        ] {
            assert!(
                Manifest::parse(&encoding_manifest(&format!("encoding={{{field}}}"), "camera")).is_err(),
                "{field}"
            );
        }
        assert!(Manifest::parse(&encoding_manifest("format='UYVY',encoding={enabled=true}", "camera")).is_err());
        assert!(Manifest::parse(&encoding_manifest("encoding={enabled=true}", "microphone")).is_err());
        let rtsp = "[[device]]\nname='synthetic'\nkind='rtsp'\nrtsp={url='rtsp://camera.invalid/video'}\nvideo={encoding={enabled=true}}\n";
        assert!(Manifest::parse(rtsp).is_err());
        assert!(Manifest::parse("[video.encoding]\ncq=52\n").is_err());
    }

    #[test]
    fn encoding_accepts_numeric_boundaries() {
        for gpu in [0, i32::MAX as u32] {
            for (cq, interval) in [(0, 100), (51, 10000)] {
                let fields = format!("encoding={{enabled=true,gpu={gpu},cq={cq},keyframe_interval_ms={interval}}}");
                assert!(Manifest::parse(&encoding_manifest(&fields, "camera")).is_ok());
            }
        }
    }

    #[test]
    fn raw_defaults_are_short_without_reducing_compressed_reader_slack() {
        let text = r#"
            [[device]]
            name = "compressed"
            kind = "camera"
            match = { name = "Synthetic compressed" }
            [[device]]
            name = "raw"
            kind = "camera"
            match = { name = "Synthetic raw" }
            video = { format = "UYVY" }
        "#;
        let linux = Manifest::parse_native(text, false).unwrap();
        assert_eq!(linux.devices[0].video().unwrap().ring_seconds, 10.0);
        assert_eq!(linux.devices[1].video().unwrap().ring_seconds, 2.0);
        let mac = Manifest::parse_native(text, true).unwrap();
        assert_eq!(mac.devices[0].video().unwrap().ring_seconds, 2.0, "AVFoundation delivers native raw");
        assert_eq!(mac.devices[1].video().unwrap().ring_seconds, 2.0);
    }

    #[test]
    fn ring_defaults_follow_effective_format_but_never_replace_explicit_horizons() {
        let text = r#"
            [video]
            format = "UYVY"
            [[device]]
            name = "raw"
            kind = "camera"
            match = { name = "Synthetic raw" }
            [[device]]
            name = "compressed"
            kind = "camera"
            match = { name = "Synthetic compressed" }
            video = { format = "MJPEG" }
        "#;
        let linux = Manifest::parse_native(text, false).unwrap();
        assert_eq!(linux.devices[0].video().unwrap().ring_seconds, 2.0);
        assert_eq!(linux.devices[1].video().unwrap().ring_seconds, 10.0);
        let explicit = text
            .replace("format = \"UYVY\"", "format = \"UYVY\"\nring_seconds = 7.0")
            .replace("video = { format = \"MJPEG\" }", "video = { format = \"MJPEG\", ring_seconds = 10.0 }");
        for native_raw in [false, true] {
            let manifest = Manifest::parse_native(&explicit, native_raw).unwrap();
            assert_eq!(manifest.devices[0].video().unwrap().ring_seconds, 7.0);
            assert_eq!(manifest.devices[1].video().unwrap().ring_seconds, 10.0);
        }
    }

    #[test]
    fn the_top_level_video_is_every_cameras_and_a_cameras_own_keys_win() {
        let m = Manifest::parse(
            r#"
            [video]
            fps = 15
            flip = true
            hold_fps = false

            [[device]]
            name = "cam0"
            kind = "camera"
            match = { usb = "0fd9:0078" }

            [[device]]
            name = "cam1"
            kind = "camera"
            match = { usb = "046d:085b" }
            video = { fps = 30, width = 1920, height = 1080 }

            [[device]]
            name = "wave3"
            kind = "microphone"
            match = { name = "Wave 3" }
            "#,
        )
        .expect("parse");
        let cam0 = m.devices[0].video().expect("camera");
        assert_eq!((cam0.width, cam0.height, cam0.fps, cam0.flip, cam0.hold_fps), (1280, 720, 15, true, false));
        let cam1 = m.devices[1].video().expect("camera");
        assert_eq!((cam1.width, cam1.height, cam1.fps, cam1.flip, cam1.hold_fps), (1920, 1080, 30, true, false));
        assert_eq!(m.devices[2].video(), None, "a microphone gets no video");
    }

    #[test]
    fn a_bad_top_level_video_is_refused() {
        assert!(Manifest::parse("[video]\nfsp = 15\n").is_err(), "a misspelled key");
        let zero =
            "[video]\nfps = 0\n\n[[device]]\nname = \"cam0\"\nkind = \"camera\"\nmatch = { usb = \"0fd9:0078\" }\n";
        assert!(Manifest::parse(zero).is_err(), "it reaches the cameras, which are checked");
    }

    const DEVBOX: &str = r#"
        [supervisor]
        settle_ms = 1000

        [[device]]
        name = "facecam"
        kind = "camera"
        match = { name = "Elgato Facecam", usb = "0fd9:0078" }
        video = { width = 1920, height = 1080, fps = 30 }
        residual_offset_ns = { video = 33000000 }

        [[device]]
        name = "wave3"
        kind = "microphone"
        match = { name = "Elgato Wave 3" }
    "#;

    #[test]
    fn the_devbox_manifest_parses_with_defaults_filled_in() {
        let m = Manifest::parse(DEVBOX).unwrap();
        assert_eq!(m.supervisor.settle_ms, 1_000);
        assert_eq!(m.supervisor.backoff_max_ms, SupervisorSettings::default().backoff_max_ms);
        let cam = &m.devices[0];
        let v = cam.video().unwrap();
        assert_eq!((v.width, v.format.as_str(), v.hold_fps), (1920, "MJPEG", true));
        assert_eq!(cam.residual_offset_ns["video"], 33_000_000);
        assert_eq!(cam.stream_name("video"), "facecam.video");
        assert_eq!(m.devices[1].audio().unwrap().batch_seconds, 1.0);
        assert!(m.devices[1].video().is_none());
        assert!(m.audio.excludes("Arducam USB Camera: Audio"));
    }

    #[test]
    fn a_manifest_that_does_not_make_sense_is_refused() {
        let dup =
            format!("{DEVBOX}\n[[device]]\nname = \"wave3\"\nkind = \"microphone\"\nmatch = {{ name = \"x\" }}\n");
        assert!(matches!(Manifest::parse(&dup), Err(ManifestError::DuplicateName(_))));
        let no_match = "[[device]]\nname = \"a\"\nkind = \"camera\"\nmatch = {}\n";
        assert!(matches!(Manifest::parse(no_match), Err(ManifestError::NoMatch(_))));
        let bad_usb = "[[device]]\nname = \"a\"\nkind = \"camera\"\nmatch = { usb = \"facecam\" }\n";
        assert!(matches!(Manifest::parse(bad_usb), Err(ManifestError::Invalid(..))));
        let bad_format =
            "[[device]]\nname = \"a\"\nkind = \"camera\"\nmatch = { name = \"x\" }\nvideo = { format = \"H264\" }\n";
        assert!(matches!(Manifest::parse(bad_format), Err(ManifestError::Invalid(..))));
        let typo = "[[device]]\nname = \"a\"\nkind = \"camera\"\nmatch = { nmae = \"x\" }\n";
        assert!(matches!(Manifest::parse(typo), Err(ManifestError::Parse(_))), "unknown fields are refused");
    }

    #[test]
    fn a_camera_carries_its_fisheye_calibration() {
        // The line tools/calibrate_fisheye.py writes, as config/devices.example.toml documents it.
        let manifest = "[[device]]\nname = \"rig\"\nkind = \"camera\"\nmatch = { name = \"x\" }\n\
            fisheye = { width = 1920, height = 1080, k = [690.0, 690.0, 959.5, 539.5], d = [0.05, -0.01, 0.002, 0.0], \
            rms = 0.4 }\n";
        let m = Manifest::parse(manifest).unwrap();
        let lens = m.devices[0].fisheye.as_ref().unwrap();
        assert_eq!((lens.width, lens.height, lens.k[2], lens.d[1], lens.rms), (1920, 1080, 959.5, -0.01, Some(0.4)));
        assert!(Manifest::parse(DEVBOX).unwrap().devices[0].fisheye.is_none());
    }

    #[test]
    fn a_malformed_fisheye_calibration_or_an_unknown_device_key_is_refused() {
        let camera = "[[device]]\nname = \"rig\"\nkind = \"camera\"\nmatch = { name = \"x\" }\n";
        for bad in [
            "fisheye = { width = 1920, height = 1080, k = [690.0, 690.0, 959.5], d = [0.0, 0.0, 0.0, 0.0] }",
            "fisheye = { width = 1920, height = 1080, k = [1.0, 1.0, 1.0, 1.0], d = [0.0, 0.0, 0.0, 0.0], kk = 1 }",
            "fisheye = { width = 1920, k = [1.0, 1.0, 1.0, 1.0], d = [0.0, 0.0, 0.0, 0.0] }",
            "pasued = true",
        ] {
            assert!(matches!(Manifest::parse(&format!("{camera}{bad}\n")), Err(ManifestError::Parse(_))), "{bad}");
        }
    }

    #[test]
    fn a_rig_camera_carries_its_mount_and_names_a_declared_rig() {
        let text = "[[rig]]\nname='pack'\n\n[[device]]\nname='front'\nkind='camera'\nmatch={name='Synthetic'}\n\
            mount={rig='pack', at=[0.0, -0.1, 0.05], turn=[30.0, -10.0, 0.0], hfov_deg=110.0}\n";
        let parsed = Manifest::parse(text).unwrap();
        assert_eq!(parsed.rigs, vec![Rig { name: "pack".into() }]);
        let mount = parsed.devices[0].mount.clone().unwrap();
        assert_eq!(mount.rig.as_deref(), Some("pack"));
        assert_eq!(mount.turn, Some([30.0, -10.0, 0.0]));
        let fixed =
            "[[device]]\nname='hall'\nkind='camera'\nmatch={name='Synthetic'}\nmount={height_m=2.1, outdoor=true}\n";
        let fixed = Manifest::parse(fixed).unwrap().devices[0].mount.clone().unwrap();
        assert_eq!((fixed.height_m, fixed.outdoor), (Some(2.1), true));
        let typo = text.replace("rig='pack'", "rig='pak'");
        assert!(Manifest::parse(&typo).is_err(), "a mount on a rig that isn't declared stops capture");
        let unknown = text.replace("hfov_deg", "hfov");
        assert!(Manifest::parse(&unknown).is_err(), "a mistyped mount key stops capture");
    }

    #[test]
    fn a_wide_camera_carries_its_lens_and_overlay_masks() {
        let manifest = "[[device]]\nname = \"bedroom\"\nkind = \"camera\"\nmatch = { name = \"x\" }\n\
            lens = { width = 896, height = 512, k = [448.0, 448.0, 447.5, 255.5], division = -0.31 }\n\
            masks = [[0.25, 0.0, 0.74, 0.09], [0.84, 0.92, 1.0, 1.0]]\n";
        let device = &Manifest::parse(manifest).unwrap().devices[0];
        let lens = device.lens.as_ref().unwrap();
        assert_eq!((lens.width, lens.k[2], lens.division), (896, 447.5, -0.31));
        assert_eq!(device.masks, vec![[0.25, 0.0, 0.74, 0.09], [0.84, 0.92, 1.0, 1.0]]);
        let typo = manifest.replace("division", "divison");
        assert!(matches!(Manifest::parse(&typo), Err(ManifestError::Parse(_))));
    }

    #[test]
    fn usb_ids_parse_as_hex() {
        assert_eq!(parse_usb("0fd9:0078"), Some((0x0fd9, 0x0078)));
        assert_eq!(parse_usb("0fd9"), None);
    }

    #[test]
    fn supervisor_timings_and_offset_keys_are_checked() {
        let dev = "[[device]]\nname = \"c\"\nkind = \"camera\"\nmatch = { name = \"x\" }\n";
        for bad in ["backoff_initial_ms = 0", "backoff_max_ms = 10", "open_timeout_ms = 0", "stop_grace_ms = 0"] {
            let text = format!("[supervisor]\n{bad}\n{dev}");
            assert!(matches!(Manifest::parse(&text), Err(ManifestError::Invalid(..))), "{bad}");
        }
        let typo = format!("{dev}residual_offset_ns = {{ vidoe = 5 }}\n");
        let err = Manifest::parse(&typo).unwrap_err().to_string();
        assert!(err.contains("vidoe"), "{err}");
        assert!(Manifest::parse(&format!("{dev}residual_offset_ns = {{ video = 5 }}\n")).is_ok());
    }
}

#[cfg(test)]
mod network_tests {
    use super::*;

    #[test]
    fn network_cameras_have_independent_virtual_nodes_and_never_debug_their_urls() {
        let manifest = Manifest::parse(
            r#"
            [[device]]
            name='entryway'
            kind='rtsp'
            rtsp={url='rtsps://user:private-password@camera.invalid:7441/private-token'}
            video={ring_seconds=10.0,flip=true}
            [[device]]
            name='workroom'
            kind='rtsp'
            rtsp={url='rtsp://camera.invalid/h264Preview_01_sub'}
        "#,
        )
        .unwrap();
        let text = format!("{manifest:?}");
        assert!(!text.contains("private-password"));
        assert!(!text.contains("private-token"));
        let nodes = crate::rtsp::configured_devices(&manifest);
        assert_eq!(nodes[0].node, "rtsp:entryway");
        assert_eq!(nodes[1].node, "rtsp:workroom");
        let supervisor = crate::supervisor::Supervisor::new(manifest);
        assert_eq!(supervisor.match_device(&nodes[0]), Some("entryway"));
        assert_eq!(supervisor.match_device(&nodes[1]), Some("workroom"));
    }

    #[test]
    fn malformed_network_config_values_and_toml_excerpts_are_redacted() {
        for text in [
            "[[device]]\nname='entryway'\nkind='rtsp'\nrtsp={url='rtsps://user:private-password@camera.invalid/private-token'",
            "[[device]]\nname='entryway'\nkind='rtsp'\nrtsp={url='rtsp://camera.invalid/private-token',transport='private-password'}",
            "[[device]]\nname='entryway'\nkind='rtsp'\nrtsp={url='https://user:private-password@camera.invalid/private-token'}",
        ] {
            let message = Manifest::parse(text).unwrap_err().to_string();
            assert!(!message.contains("private-password"));
            assert!(!message.contains("private-token"));
            assert!(!message.contains("camera.invalid"));
        }
    }
}
