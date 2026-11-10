//! A camera worker's capture loop, over a [`FrameSource`] so it's tested on synthetic frames: choosing the mode
//! (the size fallback ladder), holding the frame rate in dim light, stamping each frame, publishing it, and noticing
//! a stall or a wake. The V4L2 source is in `v4l2`.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::clock::{Anchor, Clocks, WakeWatch, take_anchor};
use crate::manifest::VideoSettings;
use crate::publish::{Capture, PublishError, Sink, VideoStream};
use crate::stamp::{StampSource, v4l2_frame};

/// A format and size a camera offers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mode {
    /// The pixel format's FourCC, like `MJPG` or `UYVY`.
    pub fourcc: [u8; 4],
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

/// The FourCC for a manifest format name: `MJPEG` is V4L2's `MJPG`; anything else is its first four bytes.
#[must_use]
pub fn fourcc(format: &str) -> [u8; 4] {
    if format.eq_ignore_ascii_case("MJPEG") {
        return *b"MJPG";
    }
    let mut out = *b"    ";
    for (o, b) in out.iter_mut().zip(format.bytes()) {
        *o = b.to_ascii_uppercase();
    }
    out
}

/// The sizes to try, in order: the one asked for, then the fallbacks.
#[must_use]
pub fn ladder(video: &VideoSettings) -> Vec<(u32, u32)> {
    let mut out = vec![(video.width, video.height)];
    for size in &video.fallback {
        if !out.contains(size) {
            out.push(*size);
        }
    }
    out
}

/// The first mode on the ladder the camera offers in the asked-for format, or `None` if it offers none of them.
#[must_use]
pub fn pick_mode(offered: &[Mode], video: &VideoSettings) -> Option<Mode> {
    let want = fourcc(&video.format);
    ladder(video).into_iter().find_map(|(width, height)| {
        offered.iter().copied().find(|m| m.fourcc == want && m.width == width && m.height == height)
    })
}

/// `V4L2_CID_EXPOSURE_AUTO`.
pub const CID_EXPOSURE_AUTO: u32 = 0x009a_0901;
/// `V4L2_CID_EXPOSURE_AUTO_PRIORITY`.
pub const CID_EXPOSURE_AUTO_PRIORITY: u32 = 0x009a_0903;
/// `V4L2_EXPOSURE_SHUTTER_PRIORITY`.
pub const EXPOSURE_SHUTTER_PRIORITY: i64 = 2;

/// Setting a camera's controls.
pub trait Controls {
    /// Sets control `id` to `value`; `true` if the camera took it.
    fn set(&mut self, id: u32, value: i64) -> bool;
}

/// Keeps a camera at its frame rate in dim light: auto exposure may not lower the frame rate, or, on a
/// camera without that control (the Elgato Facecam), shutter priority, which caps exposure and raises gain. Returns
/// what was done, for the log, or `None` if the camera has neither.
pub fn hold_frame_rate(controls: &mut dyn Controls) -> Option<&'static str> {
    if controls.set(CID_EXPOSURE_AUTO_PRIORITY, 0) {
        return Some("auto exposure may not lower the frame rate");
    }
    if controls.set(CID_EXPOSURE_AUTO, EXPOSURE_SHUTTER_PRIORITY) {
        return Some("shutter-priority exposure");
    }
    None
}

/// One frame as the driver hands it over.
#[derive(Clone, Copy, Debug)]
pub struct RawFrame<'a> {
    /// The payload (an MJPEG frame, or packed pixels).
    pub bytes: &'a [u8],
    /// The driver's timestamp, seconds.
    pub tv_sec: i64,
    /// The driver's timestamp, microseconds.
    pub tv_usec: i64,
    /// The buffer's flags (`V4L2_BUF_FLAG_*`).
    pub flags: u32,
    /// The driver's frame counter: a jump means it dropped frames.
    pub sequence: u32,
    /// How long the source says the frame was presented, nanoseconds, or zero when it doesn't say. V4L2 never
    /// says; AVFoundation does when the sample buffer carries a numeric duration.
    pub duration_ns: u64,
}

/// Where frames come from: a V4L2 device, or a script in tests.
pub trait FrameSource {
    /// Waits up to `timeout_ms` for the next frame and hands it to `on_frame` while its buffer is borrowed.
    /// `Ok(false)` on a timeout.
    ///
    /// # Errors
    ///
    /// The device's error (it was unplugged, or the stream broke).
    fn next_frame(&mut self, timeout_ms: u32, on_frame: &mut dyn FnMut(&RawFrame<'_>)) -> std::io::Result<bool>;
}

/// What opening a camera settled on.
#[derive(Clone, Debug)]
pub struct Opened {
    /// The mode it's streaming in.
    pub mode: Mode,
    /// Bytes per row (meaningful for raw formats).
    pub stride: u32,
    /// The largest frame the driver will hand over (its buffer size).
    pub max_frame_bytes: u32,
    /// Nanoseconds per frame at the rate it accepted.
    pub frame_interval_ns: u64,
    /// What holding the frame rate did, if it was asked for and the camera allowed it.
    pub hold: Option<&'static str>,
    /// The payload it delivers, as a manifest format name: `MJPEG` or `UYVY` (the stream's shape).
    pub shape: String,
}

/// A camera a worker runs: a frame source that starts streaming before the worker says it's open.
pub trait Camera: FrameSource {
    /// Starts streaming and waits for the first frame (kept for the first `next_frame`), so the camera holds its
    /// bandwidth before the next camera opens.
    ///
    /// # Errors
    ///
    /// The device's error, or a timeout.
    fn start(&mut self) -> std::io::Result<()>;

    /// Corrects what opening reported from the first frame, once [`Camera::start`] has it (AVFoundation says a
    /// frame's real size and row stride only in the frame). Nothing to do by default.
    fn refine(&self, _opened: &mut Opened) {}
}

/// What the loop is configured with, from the manifest and the mode the camera opened in.
#[derive(Clone, Copy, Debug)]
pub struct CameraConfig {
    /// The mode it opened in.
    pub mode: Mode,
    /// Bytes per row, for raw formats (0 for MJPEG).
    pub stride: u32,
    /// Nanoseconds between frames at the rate it opened at.
    pub frame_interval_ns: u64,
    /// The camera is upside down.
    pub flip: bool,
    /// No frame for this long is a stall.
    pub stall_ns: u64,
    /// The slot size: a bigger frame is dropped and counted.
    pub slot_bytes: usize,
}

/// How often a capture loop reports its counts: often enough to watch a drift run, rarely enough to read.
pub const HEALTH_NS: u64 = 10_000_000_000;

/// Why the loop returned.
#[derive(Debug)]
pub enum CameraExit {
    /// `stop` was set.
    Stopped,
    /// No frame for `stall_ns`: the worker exits, and the supervisor reopens it with backoff.
    Stalled,
    /// The host slept and woke: here is a new anchor. The worker registers again with it and runs the loop again,
    /// without closing the device.
    Woke(Anchor),
    /// The device failed (most often, it was unplugged).
    Source(std::io::Error),
    /// Publishing failed.
    Publish(PublishError),
}

/// What the loop has seen, for the worker's health line.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CameraCounts {
    /// Frames published.
    pub frames: u64,
    /// Frames the driver dropped (jumps in its sequence).
    pub driver_dropped: u64,
    /// Frames too big for a slot.
    pub oversize: u64,
    /// Frames dropped because every slot in their Tier 0 class was pinned or being written.
    pub arena_full_dropped: u64,
    /// Overwritten slots that hadn't reached Tier 1.
    pub unflushed_overwrites: u64,
    /// Frames stamped from arrival, because the driver's stamp wasn't usable.
    pub arrival_stamped: u64,
    /// Frames the driver marked damaged (`V4L2_BUF_FLAG_ERROR`), skipped.
    pub damaged: u64,
}

/// `V4L2_BUF_FLAG_ERROR`: the frame is damaged.
pub const V4L2_BUF_FLAG_ERROR: u32 = 0x0000_0040;

/// A fixed-shape destination for camera frames, optionally encoding before publication.
pub trait CameraPublisher {
    /// Publishes one frame with its original capture metadata.
    /// # Errors
    /// Encoding, arena or broker failure.
    fn publish_frame(
        &mut self,
        capture: Capture,
        bytes: &[u8],
        sink: &mut dyn Sink,
    ) -> Result<crate::publish::Published, PublishError>;
    /// The registered stream identity.
    fn stream_id(&self) -> continuum_core::StreamId;
    /// Reanchors publication after a host wake.
    fn set_anchor(&mut self, anchor: Anchor);
}

impl CameraPublisher for VideoStream {
    fn publish_frame(
        &mut self,
        capture: Capture,
        bytes: &[u8],
        sink: &mut dyn Sink,
    ) -> Result<crate::publish::Published, PublishError> {
        self.publish(capture, sink, |slot| {
            slot[..bytes.len()].copy_from_slice(bytes);
            bytes.len()
        })
    }
    fn stream_id(&self) -> continuum_core::StreamId {
        VideoStream::stream_id(self)
    }
    fn set_anchor(&mut self, anchor: Anchor) {
        VideoStream::set_anchor(self, anchor);
    }
}

/// A camera's H.264 destination. Native metadata is retained while MJPEG bytes become one H.264 access unit.
#[cfg(feature = "ffmpeg")]
pub struct EncodedPublisher {
    /// The fixed H.264 stream.
    pub stream: VideoStream,
    /// Stateful encoder with software fallback.
    pub pipeline: crate::encoding::Pipeline,
    /// Output cannot exceed its arena slot.
    pub slot_bytes: usize,
}

#[cfg(feature = "ffmpeg")]
impl CameraPublisher for EncodedPublisher {
    fn publish_frame(
        &mut self,
        capture: Capture,
        bytes: &[u8],
        sink: &mut dyn Sink,
    ) -> Result<crate::publish::Published, PublishError> {
        let encoded = self.pipeline.encode(capture, bytes).map_err(|e| PublishError::Format(e.to_string()))?;
        if encoded.len() > self.slot_bytes {
            return Err(PublishError::Format("H.264 access unit exceeds capture slot capacity".into()));
        }
        let result = self.stream.publish_encoded(Capture { stride: 0, ..capture }, &encoded, sink);
        if result.is_err() {
            self.pipeline.break_chain();
        }
        result
    }
    fn stream_id(&self) -> continuum_core::StreamId {
        self.stream.stream_id()
    }
    fn set_anchor(&mut self, anchor: Anchor) {
        self.stream.set_anchor(anchor);
    }
}

/// Runs a camera until it stops, stalls, fails, or the host wakes from sleep. `counts` carries across runs.
/// `health` is called with the counts every [`HEALTH_NS`], for the worker's log, and `offer` with each published
/// frame and its arrival time (auto exposure takes one a second, `crate::steer`).
#[allow(clippy::too_many_arguments)]
pub fn run_camera(
    source: &mut dyn FrameSource,
    stream: &mut dyn CameraPublisher,
    sink: &mut dyn Sink,
    clocks: &dyn Clocks,
    config: &CameraConfig,
    stop: &AtomicBool,
    counts: &mut CameraCounts,
    health: &mut dyn FnMut(&CameraCounts),
    offer: &mut dyn FnMut(u64, &[u8]),
) -> CameraExit {
    let mut wake = WakeWatch::new(clocks);
    // One session per call: a wake returns, and the next call's frames don't continue this one's sequence.
    let epoch = crate::publish::new_capture_epoch();
    let mut last_health = clocks.monotonic_ns();
    let mut last_frame = clocks.monotonic_ns();
    let mut last_seq: Option<u32> = None;
    let mut last_pressure_warning: Option<u64> = None;
    // Poll in short steps so `stop` and a wake are noticed promptly.
    let step_ms = u32::try_from((config.stall_ns / 1_000_000).clamp(1, 200)).unwrap_or(200);
    loop {
        if stop.load(Ordering::Relaxed) {
            return CameraExit::Stopped;
        }
        let now = clocks.monotonic_ns();
        if now.saturating_sub(last_health) >= HEALTH_NS {
            last_health = now;
            health(counts);
        }
        if wake.woke(clocks) {
            return CameraExit::Woke(take_anchor(clocks));
        }
        let mut failure = None;
        let got = source.next_frame(step_ms, &mut |raw| {
            let arrival = clocks.monotonic_ns();
            if let Some(prev) = last_seq {
                counts.driver_dropped += u64::from(raw.sequence.wrapping_sub(prev).saturating_sub(1));
            }
            last_seq = Some(raw.sequence);
            if raw.flags & V4L2_BUF_FLAG_ERROR != 0 {
                // The driver says the frame is damaged (it hands such frames over only with nodrop set).
                counts.damaged += 1;
                return;
            }
            // Only the frame itself: some UVC cameras say a whole MJPEG buffer is used, stale bytes after the image
            // and all (a 1280x720 Facecam's 900 KB buffer held a 270 KB JPEG).
            let raw = RawFrame { bytes: &raw.bytes[..payload_len(&config.mode, config.stride, raw.bytes)], ..*raw };
            let raw = &raw;
            if raw.bytes.len() > config.slot_bytes {
                counts.oversize += 1;
                return;
            }
            let stamp = v4l2_frame(raw.tv_sec, raw.tv_usec, raw.flags, config.frame_interval_ns, arrival);
            if stamp.source == StampSource::Arrival {
                counts.arrival_stamped += 1;
            }
            let capture = Capture {
                mono_ns: stamp.mono_ns,
                device_ts: stamp.device_ns,
                arrival_ns: arrival,
                device_seq: u64::from(raw.sequence),
                stride: config.stride,
                width: config.mode.width,
                height: config.mode.height,
                flip: config.flip,
                sample_rate_hz: 0,
                duration_ns: raw.duration_ns,
                capture_epoch: epoch,
            };
            let bytes = raw.bytes;
            match stream.publish_frame(capture, bytes, sink) {
                Ok(p) => {
                    counts.frames += 1;
                    counts.unflushed_overwrites += p.unflushed_overwrites as u64;
                    offer(arrival, bytes);
                }
                Err(PublishError::Shm(continuum_shm::Error::Full)) => {
                    // Full comes only from the arena claim: all slots are pinned or still being written. The
                    // device delivered this frame, so keep its sequence/time progression and do not call it a stall.
                    counts.arena_full_dropped += 1;
                    if last_pressure_warning.is_none_or(|last| arrival.saturating_sub(last) >= HEALTH_NS) {
                        last_pressure_warning = Some(arrival);
                        eprintln!("capture worker: WARN stream {}: Tier 0 slots exhausted; {} frames dropped (device sequence {}); capture continues", stream.stream_id().0, counts.arena_full_dropped, raw.sequence);
                    }
                }
                Err(e) => failure = Some(e),
            }
        });
        if let Some(e) = failure {
            return CameraExit::Publish(e);
        }
        match got {
            Ok(true) => last_frame = clocks.monotonic_ns(),
            Ok(false) => {
                if clocks.monotonic_ns().saturating_sub(last_frame) >= config.stall_ns {
                    return CameraExit::Stalled;
                }
            }
            // A signal that interrupts a wait (the supervisor's SIGTERM) is the stop it carries.
            Err(_) if stop.load(Ordering::Relaxed) => return CameraExit::Stopped,
            Err(e) => return CameraExit::Source(e),
        }
    }
}

/// How much of a driver's buffer is the frame: an MJPEG frame up to its end-of-image marker, found by walking the
/// JPEG from its start (stale bytes after it can hold an older frame's markers, so a search from the end could stop
/// in them); packed pixels, `stride * height`. A buffer that doesn't parse is kept whole.
pub fn payload_len(mode: &Mode, stride: u32, bytes: &[u8]) -> usize {
    if &mode.fourcc == b"MJPG" {
        return jpeg_len(bytes).unwrap_or(bytes.len());
    }
    let packed = stride as usize * mode.height as usize;
    if packed > 0 { bytes.len().min(packed) } else { bytes.len() }
}

/// The length of the JPEG at the start of `bytes`, through its EOI marker (`FF D9`), or `None` when it isn't one.
pub fn jpeg_len(bytes: &[u8]) -> Option<usize> {
    if bytes.get(..2)? != [0xFF, 0xD8] {
        return None;
    }
    let mut i = 2;
    loop {
        // A marker: FF (any fill FFs) then its code.
        if *bytes.get(i)? != 0xFF {
            return None;
        }
        while *bytes.get(i + 1)? == 0xFF {
            i += 1;
        }
        let code = *bytes.get(i + 1)?;
        match code {
            0xD9 => return Some(i + 2),
            0x01 | 0xD0..=0xD7 => i += 2, // no length
            _ => {
                let len = usize::from(u16::from_be_bytes([*bytes.get(i + 2)?, *bytes.get(i + 3)?]));
                if len < 2 {
                    return None;
                }
                i += 2 + len;
                if code == 0xDA {
                    // Entropy-coded data up to the next marker that isn't a stuffed FF 00 or a restart.
                    loop {
                        if *bytes.get(i)? == 0xFF {
                            let next = *bytes.get(i + 1)?;
                            if next == 0x00 || (0xD0..=0xD7).contains(&next) {
                                i += 2;
                                continue;
                            }
                            if next != 0xFF {
                                break;
                            }
                        }
                        i += 1;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_jpeg(scan: &[u8]) -> Vec<u8> {
        let mut v = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x04, 0x4A, 0x46]; // SOI, APP0 with 2 bytes of data
        v.extend([0xFF, 0xDA, 0x00, 0x03, 0x01]); // SOS with 1 byte of header
        v.extend(scan);
        v.extend([0xFF, 0xD9]);
        v
    }

    #[test]
    fn an_mjpeg_frame_ends_at_its_own_eoi_not_a_stale_one_after_it() {
        // stuffed FF 00 and a restart marker inside the scan; then a stale older frame's bytes, EOI and all
        let jpeg = tiny_jpeg(&[0x12, 0xFF, 0x00, 0x34, 0xFF, 0xD3, 0x56]);
        let mut buffer = jpeg.clone();
        buffer.extend(tiny_jpeg(&[0x99]));
        buffer.extend([0u8; 64]);
        let mjpg = Mode { fourcc: *b"MJPG", width: 1280, height: 720 };
        assert_eq!(payload_len(&mjpg, 0, &buffer), jpeg.len());
        assert_eq!(jpeg_len(&jpeg), Some(jpeg.len()));
    }

    #[test]
    fn a_buffer_that_is_not_a_whole_jpeg_is_kept_whole() {
        let mjpg = Mode { fourcc: *b"MJPG", width: 4, height: 4 };
        let cut = tiny_jpeg(&[0x12])[..9].to_vec(); // no EOI
        assert_eq!(payload_len(&mjpg, 0, &cut), cut.len());
        assert_eq!(payload_len(&mjpg, 0, &[1, 2, 3]), 3);
    }

    #[test]
    fn packed_pixels_are_stride_times_height() {
        let uyvy = Mode { fourcc: *b"UYVY", width: 4, height: 2 };
        assert_eq!(payload_len(&uyvy, 8, &[0u8; 100]), 16);
        assert_eq!(payload_len(&uyvy, 8, &[0u8; 10]), 10);
    }

    fn modes() -> Vec<Mode> {
        let m = |f: &[u8; 4], width, height| Mode { fourcc: *f, width, height };
        vec![m(b"YUYV", 1280, 720), m(b"MJPG", 1920, 1080), m(b"MJPG", 960, 540), m(b"MJPG", 640, 480)]
    }

    #[test]
    fn the_ladder_falls_back_in_order_and_in_the_asked_format() {
        let video = VideoSettings::default();
        assert_eq!(ladder(&video), vec![(1280, 720), (960, 540), (640, 480)]);
        assert_eq!(pick_mode(&modes(), &video), Some(Mode { fourcc: *b"MJPG", width: 960, height: 540 }));
        let yuyv = VideoSettings { format: "yuyv".into(), ..VideoSettings::default() };
        assert_eq!(pick_mode(&modes(), &yuyv).map(|m| (m.width, m.height)), Some((1280, 720)));
        let none = VideoSettings { format: "UYVY".into(), ..VideoSettings::default() };
        assert_eq!(pick_mode(&modes(), &none), None);
    }

    struct Cam {
        has: Vec<u32>,
        set: Vec<(u32, i64)>,
    }

    impl Controls for Cam {
        fn set(&mut self, id: u32, value: i64) -> bool {
            self.set.push((id, value));
            self.has.contains(&id)
        }
    }

    #[test]
    fn hold_frame_rate_prefers_auto_priority_then_shutter_priority() {
        let mut most = Cam { has: vec![CID_EXPOSURE_AUTO_PRIORITY, CID_EXPOSURE_AUTO], set: vec![] };
        assert_eq!(hold_frame_rate(&mut most), Some("auto exposure may not lower the frame rate"));
        assert_eq!(most.set, vec![(CID_EXPOSURE_AUTO_PRIORITY, 0)]);
        let mut facecam = Cam { has: vec![CID_EXPOSURE_AUTO], set: vec![] };
        assert_eq!(hold_frame_rate(&mut facecam), Some("shutter-priority exposure"));
        assert_eq!(facecam.set.last(), Some(&(CID_EXPOSURE_AUTO, EXPOSURE_SHUTTER_PRIORITY)));
        assert_eq!(hold_frame_rate(&mut Cam { has: vec![], set: vec![] }), None);
    }
}
