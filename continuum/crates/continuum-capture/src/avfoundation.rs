//! The AVFoundation camera source (macOS): a capture session on one camera, in the format
//! [`crate::mac_bandwidth::plan_camera`] chooses (packed 4:2:2, `2vuy`), at the manifest's frame rate. A sample buffer delegate
//! on a dispatch queue of its own copies each frame and its presentation time (on the session's clock, the host
//! clock, `mach_absolute_time`, which capture stamps with on macOS) into a bounded channel; the worker's loop reads
//! it as a [`FrameSource`]. No test opens a camera; the choices are tested in `avf_rules`.

use std::io;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use dispatch2::{DispatchQueue, DispatchRetained};
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{AnyObject, NSObject, NSObjectProtocol, ProtocolObject};
use objc2::{AllocAnyThread, DefinedClass, define_class, msg_send};
use objc2_av_foundation::{
    AVCaptureConnection, AVCaptureDevice, AVCaptureDeviceDiscoverySession, AVCaptureDeviceFormat, AVCaptureDeviceInput,
    AVCaptureDevicePosition, AVCaptureDeviceTypeBuiltInWideAngleCamera, AVCaptureDeviceTypeContinuityCamera,
    AVCaptureDeviceTypeExternal, AVCaptureOutput, AVCaptureSession, AVCaptureSessionPresetInputPriority,
    AVCaptureVideoDataOutput, AVCaptureVideoDataOutputSampleBufferDelegate, AVMediaTypeVideo,
};
use objc2_core_media::{CMSampleBuffer, CMTime, CMTimeFlags, CMVideoFormatDescriptionGetDimensions};
use objc2_core_video::{
    CVPixelBuffer, CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow, CVPixelBufferGetDataSize,
    CVPixelBufferGetHeight, CVPixelBufferGetPixelFormatType, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
    CVPixelBufferUnlockBaseAddress, kCVPixelBufferHeightKey, kCVPixelBufferPixelFormatTypeKey, kCVPixelBufferWidthKey,
    kCVPixelFormatType_422YpCbCr8,
};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString};

use crate::avf_rules::{
    CAPTURE_FOURCC, CAPTURE_SHAPE, CameraFormat, DeliveredLayout, DurationWrite, Offered, RateRange, duration_matches,
    duration_writes, fourcc_bytes, usb_from_model_id,
};
use crate::manifest::{DeviceKind, VideoSettings};
use crate::stamp::{V4L2_TIMESTAMP_MONOTONIC, V4L2_TSTAMP_SRC_SOE, cmtime_ns, sample_duration_ns};
use crate::supervisor::DeviceInfo;
use crate::video::{Camera, FrameSource, Mode, Opened, RawFrame};

/// The key prefix for an AVFoundation camera; the rest is its `uniqueID`.
pub const KEY_PREFIX: &str = "avfoundation:";

/// Frames that may wait for the worker; the delegate drops (and counts) any beyond.
const FRAME_DEPTH: usize = 8;

/// The first frame after the session starts can take a while; longer than this is a failed open.
pub const FIRST_FRAME_TIMEOUT: Duration = Duration::from_secs(10);

fn discovery() -> Retained<NSArray<AVCaptureDevice>> {
    // SAFETY: the device type and media type statics are AVFoundation's own constants.
    unsafe {
        let types = NSArray::from_slice(&[
            AVCaptureDeviceTypeExternal,
            AVCaptureDeviceTypeBuiltInWideAngleCamera,
            AVCaptureDeviceTypeContinuityCamera,
        ]);
        AVCaptureDeviceDiscoverySession::discoverySessionWithDeviceTypes_mediaType_position(
            &types,
            AVMediaTypeVideo,
            AVCaptureDevicePosition::Unspecified,
        )
        .devices()
    }
}

/// Every camera AVFoundation has now, as hotplug devices: keyed and opened by `uniqueID`, named by its localized
/// name, with USB ids from a UVC camera's model id. It drains its own autorelease pool, so a thread that calls it
/// every second doesn't collect what AVFoundation autoreleases.
#[must_use]
pub fn cameras() -> Vec<DeviceInfo> {
    autoreleasepool(|_| cameras_now())
}

fn cameras_now() -> Vec<DeviceInfo> {
    discovery()
        .iter()
        .map(|d| {
            // SAFETY: plain property reads on a device the discovery session returned.
            let (id, name, model) =
                unsafe { (d.uniqueID().to_string(), d.localizedName().to_string(), d.modelID().to_string()) };
            DeviceInfo {
                key: format!("{KEY_PREFIX}{id}"),
                kind: DeviceKind::Camera,
                name,
                usb: usb_from_model_id(&model),
                node: id.clone(),
                paths: vec![id],
            }
        })
        .collect()
}

/// A frame the delegate copied out of a sample buffer, with its real size and row stride.
struct OwnedFrame {
    bytes: Vec<u8>,
    pts_ns: u64,
    duration_ns: u64,
    sequence: u32,
    width: u32,
    height: u32,
    bytes_per_row: u32,
    fourcc: [u8; 4],
}

struct Ivars {
    frames: SyncSender<Result<OwnedFrame, String>>,
    layout: Arc<Mutex<DeliveredLayout>>,
    fatal: Arc<Mutex<Option<String>>>,
    sequence: AtomicU32,
    dropped: Arc<AtomicU64>,
}

define_class!(
    // SAFETY: NSObject has no subclassing requirements, and this class doesn't implement Drop.
    #[unsafe(super(NSObject))]
    #[name = "ContinuumCaptureFrameDelegate"]
    #[ivars = Ivars]
    struct FrameDelegate;

    unsafe impl NSObjectProtocol for FrameDelegate {}

    unsafe impl AVCaptureVideoDataOutputSampleBufferDelegate for FrameDelegate {
        #[unsafe(method(captureOutput:didOutputSampleBuffer:fromConnection:))]
        fn did_output(&self, _output: &AVCaptureOutput, sample: &CMSampleBuffer, _connection: &AVCaptureConnection) {
            let ivars = self.ivars();
            // Every frame counts, delivered or not, so a jump in the sequence is a frame lost.
            let sequence = ivars.sequence.fetch_add(1, Ordering::Relaxed);
            // SAFETY: the sample buffer is valid for the call; its pixel buffer is locked read-only while copied.
            let frame = unsafe {
                let time = sample.presentation_time_stamp();
                let Some(pts_ns) = cmtime_ns(time.value, time.timescale) else { return };
                let span = sample.duration();
                let numeric = span.flags.contains(CMTimeFlags::Valid)
                    && !span.flags.intersects(CMTimeFlags::ImpliedValueFlagsMask);
                let duration_ns = sample_duration_ns(span.value, span.timescale, numeric);
                let Some(pixels) = sample.image_buffer() else { return };
                let row = CVPixelBufferGetBytesPerRow(&pixels);
                let rows = CVPixelBufferGetHeight(&pixels);
                let delivered = Mode {
                    fourcc: fourcc_bytes(CVPixelBufferGetPixelFormatType(&pixels)),
                    width: u32::try_from(objc2_core_video::CVPixelBufferGetWidth(&pixels)).unwrap_or(0),
                    height: u32::try_from(rows).unwrap_or(0),
                };
                let Some(len) = row.checked_mul(rows) else {
                    self.fail("native frame length overflow".into());
                    return;
                };
                let stride = u32::try_from(row).unwrap_or(0);
                let validation = ivars
                    .layout
                    .lock()
                    .map_err(|_| "capture layout lock was poisoned".to_owned())
                    .and_then(|mut layout| layout.observe(delivered, stride, len));
                if let Err(error) = validation {
                    self.fail(error);
                    return;
                }
                if len > CVPixelBufferGetDataSize(&pixels) {
                    self.fail("native frame is shorter than its row layout".into());
                    return;
                }
                if CVPixelBufferLockBaseAddress(&pixels, CVPixelBufferLockFlags::ReadOnly) != 0 {
                    self.fail("cannot lock delivered native pixel buffer".into());
                    return;
                }
                let _locked = PixelLock(&pixels);
                let base = CVPixelBufferGetBaseAddress(&pixels);
                if base.is_null() {
                    self.fail("delivered native pixel buffer has no bytes".into());
                    return;
                }
                let bytes = std::slice::from_raw_parts(base.cast::<u8>(), len).to_vec();
                OwnedFrame {
                    bytes,
                    pts_ns,
                    duration_ns,
                    sequence,
                    width: delivered.width,
                    height: delivered.height,
                    bytes_per_row: stride,
                    fourcc: delivered.fourcc,
                }
            };
            if frame.bytes.is_empty() {
                return;
            }
            if let Err(TrySendError::Full(_)) = ivars.frames.try_send(Ok(frame)) {
                ivars.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }

        #[unsafe(method(captureOutput:didDropSampleBuffer:fromConnection:))]
        fn did_drop(&self, _output: &AVCaptureOutput, _sample: &CMSampleBuffer, _connection: &AVCaptureConnection) {
            // AVFoundation dropped it (the queue was late): the sequence skips it.
            self.ivars().sequence.fetch_add(1, Ordering::Relaxed);
        }
    }
);

impl FrameDelegate {
    fn fail(&self, error: String) {
        if let Ok(mut fatal) = self.ivars().fatal.lock() {
            fatal.get_or_insert_with(|| error.clone());
        }
        let _ = self.ivars().frames.try_send(Err(error));
    }

    fn new(ivars: Ivars) -> Retained<Self> {
        let this = Self::alloc().set_ivars(ivars);
        // SAFETY: NSObject's init on a freshly allocated object.
        unsafe { msg_send![super(this), init] }
    }
}

/// An AVFoundation camera, streaming.
pub struct AvfSource {
    resources: SessionResources,
    _delegate: Retained<FrameDelegate>,
    _queue: DispatchRetained<DispatchQueue>,
    frames: Receiver<Result<OwnedFrame, String>>,
    first: Option<OwnedFrame>,
    configured: Mode,
    layout: Arc<Mutex<DeliveredLayout>>,
    frame_interval_ns: u64,
    fatal: Arc<Mutex<Option<String>>>,
    dropped: Arc<AtomicU64>,
}

/// Configuration guards always commit before a session can be stopped or torn down.
struct Configuration(Retained<AVCaptureSession>);
impl Configuration {
    unsafe fn begin(session: &Retained<AVCaptureSession>) -> Self {
        // SAFETY: a worker owns its session exclusively, and never nests configuration guards.
        unsafe {
            session.beginConfiguration();
            Self(session.clone())
        }
    }
}
impl Drop for Configuration {
    fn drop(&mut self) {
        // SAFETY: this guard owns exactly one successful beginConfiguration, and commits before session teardown.
        unsafe {
            self.0.commitConfiguration();
        }
    }
}

struct DeviceLock<'a>(&'a AVCaptureDevice);
impl Drop for DeviceLock<'_> {
    fn drop(&mut self) {
        // SAFETY: this guard is constructed only after lockForConfiguration succeeds on this live device.
        unsafe {
            self.0.unlockForConfiguration();
        }
    }
}
struct PixelLock<'a>(&'a CVPixelBuffer);
impl Drop for PixelLock<'_> {
    fn drop(&mut self) {
        // SAFETY: this buffer remains retained by the callback, and the guard matches its successful read-only lock.
        unsafe {
            CVPixelBufferUnlockBaseAddress(self.0, CVPixelBufferLockFlags::ReadOnly);
        }
    }
}

/// Constructed before any input/output is attached, so every setup error releases partial session ownership.
struct SessionResources {
    session: Retained<AVCaptureSession>,
    output: Retained<AVCaptureVideoDataOutput>,
    queue: DispatchRetained<DispatchQueue>,
    closed: bool,
}
impl SessionResources {
    fn close(&mut self) {
        if std::mem::replace(&mut self.closed, true) {
            return;
        }
        use crate::teardown::step;
        // Each step is timed while a worker closes its camera: stopRunning is synchronous, so its return is
        // the session's own completion, and a hang shows in the log as the step a kill cut off.
        // SAFETY: all configuration guards have left scope. Stop delivery before detaching the delegate and
        // removing this session's inputs/outputs; the AvfSource still retains delegate/queue during teardown.
        unsafe {
            step("session stopRunning", || self.session.stopRunning());
            step("detach frame delegate", || self.output.setSampleBufferDelegate_queue(None, None));
            // This method runs only on the worker thread, never on the delegate queue. Drain callbacks that
            // were already queued before releasing the source's retained delegate and buffers.
            step("drain delegate queue", || self.queue.exec_sync(|| {}));
            step("remove session inputs and outputs", || {
                let _configuration = Configuration::begin(&self.session);
                let outputs: Vec<_> = self.session.outputs().iter().collect();
                let inputs: Vec<_> = self.session.inputs().iter().collect();
                for output in outputs {
                    self.session.removeOutput(&output);
                }
                for input in inputs {
                    self.session.removeInput(&input);
                }
            });
        }
    }
}
impl Drop for SessionResources {
    fn drop(&mut self) {
        self.close();
    }
}

/// The formats a camera offers, with the highest frame rate and the rate ranges of each.
fn offered(device: &AVCaptureDevice) -> Vec<(Offered, Vec<RateRange>, Retained<AVCaptureDeviceFormat>)> {
    // SAFETY: plain property reads on the device's own formats.
    unsafe {
        device
            .formats()
            .iter()
            .map(|f| {
                let desc = f.formatDescription();
                let dims = CMVideoFormatDescriptionGetDimensions(&desc);
                let ranges: Vec<RateRange> = f
                    .videoSupportedFrameRateRanges()
                    .iter()
                    .map(|r| {
                        let (lo, hi) = (r.minFrameDuration(), r.maxFrameDuration());
                        RateRange {
                            min_fps: r.minFrameRate(),
                            max_fps: r.maxFrameRate(),
                            min_duration: (lo.value, lo.timescale),
                            max_duration: (hi.value, hi.timescale),
                        }
                    })
                    .collect();
                let max_fps = ranges.iter().map(|r| r.max_fps).fold(0.0, f64::max);
                let mode = Mode {
                    fourcc: fourcc_bytes(desc.media_sub_type()),
                    width: u32::try_from(dims.width).unwrap_or(0),
                    height: u32::try_from(dims.height).unwrap_or(0),
                };
                (Offered { mode, max_fps }, ranges, f)
            })
            .collect()
    }
}

/// Reads a camera's actual native format/rate inventory without creating a session or opening/configuring it.
/// # Errors
/// NotFound when the explicit AVFoundation uniqueID is no longer present.
pub fn camera_inventory(node: &str) -> io::Result<Vec<CameraFormat>> {
    autoreleasepool(|_| {
        // SAFETY: these are read-only discovery/format properties; no session or configuration lock is created.
        unsafe {
            let device = AVCaptureDevice::deviceWithUniqueID(&NSString::from_str(node))
                .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("no camera {node}")))?;
            Ok(offered(&device).into_iter().map(|(offered, rates, _)| CameraFormat { offered, rates }).collect())
        }
    })
}

fn rate_ranges(format: &AVCaptureDeviceFormat) -> Vec<RateRange> {
    // SAFETY: read-only properties of a retained native format returned by this device's inventory.
    unsafe {
        format
            .videoSupportedFrameRateRanges()
            .iter()
            .map(|range| {
                let minimum = range.minFrameDuration();
                let maximum = range.maxFrameDuration();
                RateRange {
                    min_fps: range.minFrameRate(),
                    max_fps: range.maxFrameRate(),
                    min_duration: (minimum.value, minimum.timescale),
                    max_duration: (maximum.value, maximum.timescale),
                }
            })
            .collect()
    }
}

unsafe fn apply_format(
    device: &AVCaptureDevice,
    format: &AVCaptureDeviceFormat,
    output: &AVCaptureVideoDataOutput,
    width: u32,
    height: u32,
    duration: CMTime,
    hold_fps: bool,
) -> io::Result<()> {
    // SAFETY: the format belongs to this device, and duration was selected from its exact supported ranges.
    unsafe {
        device.lockForConfiguration().map_err(|error| io::Error::other(error.localizedDescription().to_string()))?;
        let _lock = DeviceLock(device);
        device.setActiveFormat(format);
        let ranges = rate_ranges(format);
        let writes = duration_writes(&ranges, (duration.value, duration.timescale), hold_fps)
            .ok_or_else(|| io::Error::other("native frame-duration limits are unsupported"))?;
        for write in writes {
            let (DurationWrite::Minimum(time) | DurationWrite::Maximum(time)) = write;
            let time = CMTime { value: time.0, timescale: time.1, flags: CMTimeFlags::Valid, epoch: 0 };
            match write {
                DurationWrite::Minimum(_) => device.setActiveVideoMinFrameDuration(time),
                DurationWrite::Maximum(_) => device.setActiveVideoMaxFrameDuration(time),
            }
        }
        let keys: [&NSString; 3] = [
            &*(kCVPixelBufferPixelFormatTypeKey as *const _ as *const NSString),
            &*(kCVPixelBufferWidthKey as *const _ as *const NSString),
            &*(kCVPixelBufferHeightKey as *const _ as *const NSString),
        ];
        let pixel = NSNumber::new_u32(kCVPixelFormatType_422YpCbCr8);
        let width = NSNumber::new_u32(width);
        let height = NSNumber::new_u32(height);
        let settings =
            NSDictionary::<NSString, AnyObject>::from_slices(&keys, &[pixel.as_ref(), width.as_ref(), height.as_ref()]);
        output.setVideoSettings(Some(&settings));
    }
    Ok(())
}

/// Why AVFoundation didn't keep the chosen format and rate after commit/start, or `None` if it did. The preset is
/// not required to read back as inputPriority: on macOS 27 it reads otherwise even while the device keeps the exact
/// activeFormat and frame durations, and those are what capture depends on.
unsafe fn format_lost(
    session: &AVCaptureSession,
    device: &AVCaptureDevice,
    format: &AVCaptureDeviceFormat,
    duration: CMTime,
    hold_fps: bool,
) -> Option<String> {
    // SAFETY: plain property reads on this session's device after commit/start.
    unsafe {
        let actual = device.activeFormat();
        let minimum = device.activeVideoMinFrameDuration();
        let maximum = device.activeVideoMaxFrameDuration();
        let exact = |t: CMTime, wanted: (i64, i32)| {
            t.flags.contains(CMTimeFlags::Valid)
                && !t.flags.intersects(CMTimeFlags::ImpliedValueFlagsMask)
                && t.epoch == 0
                && duration_matches((t.value, t.timescale), wanted)
        };
        if !format.isEqual(Some(actual.as_ref())) {
            let dims = CMVideoFormatDescriptionGetDimensions(&actual.formatDescription());
            let (w, h) = (dims.width, dims.height);
            return Some(format!("activeFormat is {w}x{h}, not the chosen format"));
        }
        let want = (duration.value, duration.timescale);
        if !exact(minimum, want) {
            let got = (minimum.value, minimum.timescale);
            return Some(format!("min frame duration is {got:?}, not {want:?}"));
        }
        let Some(writes) = duration_writes(&rate_ranges(format), want, hold_fps) else {
            return Some("the native format no longer has valid duration limits".into());
        };
        let Some(wanted_max) = writes.iter().rev().find_map(|write| match write {
            DurationWrite::Maximum(time) => Some(*time),
            DurationWrite::Minimum(_) => None,
        }) else {
            return Some("the native duration policy has no maximum".into());
        };
        if !exact(maximum, wanted_max) {
            let got = (maximum.value, maximum.timescale);
            return Some(format!("max frame duration is {got:?}, not {wanted_max:?}"));
        }
        let _ = session;
        None
    }
}

impl AvfSource {
    /// Opens the camera whose `uniqueID` is `node` in the chosen format and frame rate, holding the rate if asked
    /// (both frame duration limits pinned, so the camera can't slow down in dim light). The session runs once
    /// this returns; [`Camera::start`] waits for its first frame.
    ///
    /// # Errors
    ///
    /// `NotFound` for no such camera, `Unsupported` for no usable format, or AVFoundation's error.
    pub fn open(node: &str, video: &VideoSettings) -> io::Result<(Self, Opened)> {
        Self::open_with_share(node, video, u64::MAX)
    }

    /// Opens only a native format/rate admitted by this camera's authorized fixed bandwidth share.
    /// # Errors
    /// Discovery, admission, configuration/readback, or unsupported output failures, as with [`Self::open`].
    pub fn open_with_share(
        node: &str,
        video: &VideoSettings,
        share_bytes_per_second: u64,
    ) -> io::Result<(Self, Opened)> {
        autoreleasepool(|_| Self::open_in_pool(node, video, share_bytes_per_second))
    }

    fn open_in_pool(node: &str, video: &VideoSettings, share_bytes_per_second: u64) -> io::Result<(Self, Opened)> {
        let other = |e: Retained<objc2_foundation::NSError>| io::Error::other(e.localizedDescription().to_string());
        // SAFETY: this worker owns the new session. Input/output topology is installed before native format,
        // supported durations, and explicit output settings, under configuration and device-lock guards. Guards
        // unwind in reverse order, so commit/unlock precede stop/detach/removal on every ordinary error path.
        unsafe {
            let device = AVCaptureDevice::deviceWithUniqueID(&NSString::from_str(node))
                .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("no camera {node}")))?;
            let formats = offered(&device);
            let inventory: Vec<CameraFormat> = formats
                .iter()
                .map(|(offered, rates, _)| CameraFormat { offered: *offered, rates: rates.clone() })
                .collect();
            // The worker has already fitted these effective settings to its share. Use the identical selector
            // again against current inventory, so native format/rate cannot diverge from admission's choice.
            let plan = crate::mac_bandwidth::plan_camera(node, video, &inventory, share_bytes_per_second)
                .map_err(io::Error::other)?;
            let choice = plan.native.offered;
            let format = formats
                .iter()
                .find(|(offered, rates, _)| *offered == choice && *rates == plan.native.rates)
                .map(|(_, _, format)| format.clone())
                .ok_or_else(|| io::Error::other("planned camera format vanished"))?;
            let (value, timescale) = plan.frame_duration;
            let duration = CMTime { value, timescale, flags: CMTimeFlags::Valid, epoch: 0 };

            let (width, height) = (choice.mode.width, choice.mode.height);
            let expected = Mode { fourcc: CAPTURE_FOURCC, width, height };
            let layout = Arc::new(Mutex::new(DeliveredLayout::new(
                expected,
                plan.frame_duration,
                share_bytes_per_second,
                plan.bytes_per_second,
            )));
            let (tx, frames) = sync_channel(FRAME_DEPTH);
            let dropped = Arc::new(AtomicU64::new(0));
            let fatal = Arc::new(Mutex::new(None));
            let delegate = FrameDelegate::new(Ivars {
                frames: tx,
                layout: Arc::clone(&layout),
                fatal: Arc::clone(&fatal),
                sequence: AtomicU32::new(0),
                dropped: Arc::clone(&dropped),
            });
            let queue = DispatchQueue::new("continuum.capture.frames", None);
            let resources = SessionResources {
                session: AVCaptureSession::new(),
                output: AVCaptureVideoDataOutput::new(),
                queue: queue.clone(),
                closed: false,
            };
            {
                let _configuration = Configuration::begin(&resources.session);
                let input = AVCaptureDeviceInput::deviceInputWithDevice_error(&device).map_err(other)?;
                if !resources.session.canAddInput(&input) {
                    return Err(io::Error::other("the session won't take the camera"));
                }
                resources.session.addInput(&input);
                if !resources.session.canAddOutput(&resources.output) {
                    return Err(io::Error::other("the session won't take the video output"));
                }
                resources.session.addOutput(&resources.output);
                // macOS often answers canSetSessionPreset(inputPriority) = false (seen on macOS 27 with three Arducam
                // UVC cameras), yet setting activeFormat on a device in the session switches the preset to
                // inputPriority by itself. Ask for it when allowed; format_lost verifies the format and frame durations after
                // start either way, so a camera that really can't keep the format still fails there.
                if resources.session.canSetSessionPreset(AVCaptureSessionPresetInputPriority) {
                    resources.session.setSessionPreset(AVCaptureSessionPresetInputPriority);
                }
                resources.output.setAlwaysDiscardsLateVideoFrames(true);
                resources.output.setAutomaticallyConfiguresOutputBufferDimensions(false);
                resources.output.setDeliversPreviewSizedOutputBuffers(false);
                apply_format(&device, &format, &resources.output, width, height, duration, video.hold_fps)?;
            }
            resources.session.startRunning();
            if format_lost(&resources.session, &device, &format, duration, video.hold_fps).is_some() {
                // startRunning can reset the frame durations to the format's default (1/30 on these Arducams). Set them
                // again on the running device under its own lock only: a session begin/commit here re-applies the
                // preset and resets them once more.
                apply_format(&device, &format, &resources.output, width, height, duration, video.hold_fps)?;
                std::thread::sleep(Duration::from_millis(200));
                if let Some(why) = format_lost(&resources.session, &device, &format, duration, video.hold_fps) {
                    return Err(io::Error::other(format!(
                        "AVFoundation did not retain the requested native format/frame duration after commit/start: {why}"
                    )));
                }
            }
            resources.output.setSampleBufferDelegate_queue(Some(ProtocolObject::from_ref(&*delegate)), Some(&queue));
            let hold = video.hold_fps.then_some("frame duration pinned and verified");
            let stride = width
                .checked_mul(2)
                .and_then(|row| row.checked_add(63))
                .map(|row| row / 64 * 64)
                .ok_or_else(|| io::Error::other("capture stride overflow"))?;
            let opened = Opened {
                mode: Mode { fourcc: CAPTURE_FOURCC, width, height },
                stride,
                max_frame_bytes: stride.checked_mul(height).ok_or_else(|| io::Error::other("capture size overflow"))?,
                frame_interval_ns: u64::try_from(i128::from(value) * 1_000_000_000 / i128::from(timescale))
                    .ok()
                    .filter(|interval| *interval > 0)
                    .ok_or_else(|| io::Error::other("selected frame duration cannot be represented in nanoseconds"))?,
                hold,
                shape: CAPTURE_SHAPE.into(),
            };
            Ok((
                Self {
                    resources,
                    _delegate: delegate,
                    _queue: queue,
                    frames,
                    first: None,
                    configured: expected,
                    layout,
                    frame_interval_ns: opened.frame_interval_ns,
                    fatal,
                    dropped,
                },
                opened,
            ))
        }
    }

    /// Frames the delegate dropped because the worker fell behind.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

impl AvfSource {
    fn failure(&mut self, message: impl Into<String>) -> io::Error {
        self.resources.close();
        io::Error::other(message.into())
    }
    fn check_fatal(&mut self) -> io::Result<()> {
        let error = self.fatal.lock().ok().and_then(|error| error.clone());
        if let Some(error) = error {
            return Err(self.failure(error));
        }
        Ok(())
    }
    fn checked(&mut self, frame: Result<OwnedFrame, String>) -> io::Result<OwnedFrame> {
        self.check_fatal()?;
        let frame = frame.map_err(|error| self.failure(error))?;
        let delivered = Mode { fourcc: frame.fourcc, width: frame.width, height: frame.height };
        let validation = self
            .layout
            .lock()
            .map_err(|_| "capture layout lock was poisoned".to_owned())
            .and_then(|mut layout| layout.observe(delivered, frame.bytes_per_row, frame.bytes.len()));
        validation.map_err(|error| self.failure(error))?;
        Ok(frame)
    }
}

impl Drop for AvfSource {
    fn drop(&mut self) {
        self.resources.close();
    }
}

fn raw(frame: &OwnedFrame) -> RawFrame<'_> {
    // The presentation time is the host clock's: hand it over the way a monotonic start-of-exposure V4L2 stamp
    // is, so the loop takes it as is.
    RawFrame {
        bytes: &frame.bytes,
        tv_sec: i64::try_from(frame.pts_ns / 1_000_000_000).unwrap_or(0),
        tv_usec: i64::try_from(frame.pts_ns % 1_000_000_000 / 1_000).unwrap_or(0),
        flags: V4L2_TIMESTAMP_MONOTONIC | V4L2_TSTAMP_SRC_SOE,
        sequence: frame.sequence,
        duration_ns: frame.duration_ns,
    }
}

impl FrameSource for AvfSource {
    fn next_frame(&mut self, timeout_ms: u32, on_frame: &mut dyn FnMut(&RawFrame<'_>)) -> io::Result<bool> {
        self.check_fatal()?;
        if let Some(first) = self.first.take() {
            on_frame(&raw(&first));
            return Ok(true);
        }
        match self.frames.recv_timeout(Duration::from_millis(u64::from(timeout_ms))) {
            Ok(frame) => {
                let frame = self.checked(frame)?;
                on_frame(&raw(&frame));
                Ok(true)
            }
            Err(RecvTimeoutError::Timeout) => Ok(false),
            Err(RecvTimeoutError::Disconnected) => Err(self.failure("the capture session ended")),
        }
    }
}

impl Camera for AvfSource {
    fn start(&mut self) -> io::Result<()> {
        self.check_fatal()?;
        if self.first.is_none() {
            let frame = self
                .frames
                .recv_timeout(FIRST_FRAME_TIMEOUT)
                .map_err(|_| self.failure("the camera sent no first frame"))?;
            let frame = self.checked(frame)?;
            eprintln!(
                "capture worker: configured AVFoundation {}x{}, verified first output {}x{} stride {} {:?}, verified frame duration {} ns (rate 1000000000/{} fps); native admission unchanged",
                self.configured.width,
                self.configured.height,
                frame.width,
                frame.height,
                frame.bytes_per_row,
                frame.fourcc,
                self.frame_interval_ns,
                self.frame_interval_ns
            );
            self.first = Some(frame);
        }
        Ok(())
    }

    /// The first frame's real size and row stride, and room for a frame of that size.
    fn refine(&self, opened: &mut Opened) {
        let Some(f) = &self.first else { return };
        opened.mode = Mode { fourcc: f.fourcc, width: f.width, height: f.height };
        opened.stride = f.bytes_per_row;
        opened.max_frame_bytes = u32::try_from(f.bytes.len()).unwrap_or(u32::MAX);
    }
}
