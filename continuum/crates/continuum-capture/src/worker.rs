//! One capture worker: one physical device, in its own process (docs/CONTINUUM.md, "One process per physical
//! device"). It registers its stream with the broker, opens the device, says `opened` on stdout (the supervisor's
//! cue to hand the open token on), and captures until it's told to stop, the device goes or stalls, or the broker
//! goes. After a wake it registers again with a new anchor and carries on without closing the device.
//!
//! Everything goes into the broker's Tier 0 arena, mapped from `Welcome`: a camera's frames into slots of the smallest
//! class that holds one, a microphone's blocks into the ring the broker carved for its stream at registration.

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use continuum_client::{Client, ClientOptions};
use continuum_core::{PayloadFormat, StreamId};

use crate::audio::Microphone;
use crate::clock::{Clocks, SystemClocks, take_anchor};
use crate::manifest::{Device, DeviceKind, Manifest};
use crate::publish::{AudioStream, StreamMeta, VideoStream};
use crate::register::PcmFormat;
use crate::video::{Camera, Opened};

/// What a worker is told to do.
#[derive(Clone, Debug)]
pub struct WorkerArgs {
    /// The device manifest.
    pub manifest: PathBuf,
    /// The manifest entry this worker serves.
    pub device: String,
    /// The node to open: `/dev/video0`, or a PipeWire node name.
    pub node: String,
    /// The broker's socket.
    pub socket: PathBuf,
    /// The supervisor's fixed per-camera raw bandwidth share on macOS, in bytes per second.
    pub mac_raw_share: Option<u64>,
}

/// How a worker ended, as its exit code: the supervisor restarts it with backoff whatever the reason, and the code
/// is for the log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum WorkerEnd {
    /// It was told to stop.
    Stopped = 0,
    /// Its arguments or the manifest were wrong: restarting won't help, but the supervisor backs off anyway.
    BadSetup = 2,
    /// The device couldn't be opened, failed, or went away.
    Device = 3,
    /// The device stopped sending.
    Stalled = 4,
    /// The broker refused or went away.
    Broker = 5,
    /// Shared memory failed.
    Memory = 6,
}

/// Parses `--manifest P --device NAME --node NODE --socket S`.
///
/// # Errors
///
/// What was missing or unknown.
pub fn parse_args(args: impl IntoIterator<Item = String>) -> Result<WorkerArgs, String> {
    let (mut manifest, mut device, mut node, mut socket) = (None, None, None, None);
    let mut mac_raw_share = None;
    let mut it = args.into_iter();
    while let Some(flag) = it.next() {
        let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--manifest" => manifest = Some(PathBuf::from(value)),
            "--device" => device = Some(value),
            "--node" => node = Some(value),
            "--socket" => socket = Some(PathBuf::from(value)),
            "--mac-raw-share" => {
                mac_raw_share = Some(
                    value
                        .parse::<u64>()
                        .ok()
                        .filter(|v| *v > 0)
                        .ok_or("--mac-raw-share needs positive bytes per second")?,
                );
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(WorkerArgs {
        manifest: manifest.ok_or("--manifest is required")?,
        device: device.ok_or("--device is required")?,
        node: node.ok_or("--node is required")?,
        socket: socket.ok_or("--socket is required")?,
        mac_raw_share,
    })
}

/// Tells the supervisor the device is open: it may start the next worker.
fn say_opened() {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "opened");
    let _ = out.flush();
}

/// Runs a worker to its end.
pub fn run(args: &WorkerArgs, stop: &AtomicBool) -> WorkerEnd {
    let text = match std::fs::read_to_string(&args.manifest) {
        Ok(t) => t,
        Err(e) => {
            tracing::error!("can't read {}: {e}", args.manifest.display());
            return WorkerEnd::BadSetup;
        }
    };
    let manifest = match Manifest::parse(&text) {
        Ok(m) => m,
        Err(e) => {
            tracing::error!("{e}");
            return WorkerEnd::BadSetup;
        }
    };
    let Some(device) = manifest.devices.iter().find(|d| d.enabled && d.name == args.device).cloned() else {
        tracing::error!("no device {:?} in the manifest", args.device);
        return WorkerEnd::BadSetup;
    };
    #[cfg(target_os = "macos")]
    let (device, mac_raw_share) = match plan_mac_camera(args, &manifest, device) {
        Ok(planned) => planned,
        Err(error) => {
            tracing::error!("macOS bandwidth admission: {error}");
            return WorkerEnd::BadSetup;
        }
    };
    #[cfg(not(target_os = "macos"))]
    let mac_raw_share = None;
    let rt = match tokio::runtime::Builder::new_multi_thread().worker_threads(1).enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            tracing::error!("no runtime: {e}");
            return WorkerEnd::BadSetup;
        }
    };
    let mut client =
        match rt.block_on(Client::connect(&args.socket, ClientOptions::new(format!("capture {}", device.name)))) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("can't reach the broker at {}: {e}", args.socket.display());
                return WorkerEnd::Broker;
            }
        };
    let clocks = match SystemClocks::new() {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("no boot id: {e}");
            return WorkerEnd::BadSetup;
        }
    };
    let handle = rt.handle().clone();
    let end = match device.kind {
        DeviceKind::Camera => match open_camera_with_share(&args.node, &device, mac_raw_share) {
            Ok((mut source, opened)) => {
                say_opened();
                let end = camera(
                    &handle,
                    &mut client,
                    &device,
                    (&args.node, &args.socket, &args.manifest),
                    source.as_mut(),
                    &opened,
                    &clocks,
                    stop,
                );
                // Release the device first, timed and logged, before anything else can hold up the exit (#1388).
                crate::teardown::close(&device.name, || drop(source));
                end
            }
            Err(e) => {
                tracing::warn!("{}: can't open {}: {e}", device.name, args.node);
                WorkerEnd::Device
            }
        },
        DeviceKind::Rtsp => {
            #[cfg(feature = "ffmpeg")]
            {
                crate::rtsp::run(&handle, &mut client, &device, &clocks, stop, say_opened)
            }
            #[cfg(not(feature = "ffmpeg"))]
            {
                tracing::error!("{}: RTSP requires the ffmpeg build feature", device.name);
                WorkerEnd::BadSetup
            }
        }
        DeviceKind::Microphone | DeviceKind::Xvf3800 => match open_microphone(&args.node) {
            Ok((mut source, pcm)) => {
                say_opened();
                let end = microphone(&handle, &mut client, &device, source.as_mut(), pcm, &clocks, stop);
                crate::teardown::close(&device.name, || drop(source));
                end
            }
            Err(e) => {
                tracing::warn!("{}: can't open {}: {e}", device.name, args.node);
                WorkerEnd::Device
            }
        },
        DeviceKind::Teensy => {
            tracing::error!("the Teensy worker isn't built yet");
            WorkerEnd::BadSetup
        }
    };
    let _ = rt.block_on(client.close("capture worker ended"));
    end
}

#[cfg(target_os = "macos")]
fn plan_mac_camera(
    args: &WorkerArgs,
    manifest: &Manifest,
    mut device: Device,
) -> Result<(Device, Option<u64>), String> {
    if device.kind != DeviceKind::Camera {
        return Ok((device, None));
    }
    let share = args.mac_raw_share.map_or_else(|| crate::mac_bandwidth::manifest_share(manifest), Ok)?;
    let inventory = crate::avfoundation::camera_inventory(&args.node).map_err(|error| error.to_string())?;
    let requested = device.video().ok_or("camera has no video settings")?;
    let chosen = crate::mac_bandwidth::plan_camera(&device.name, &requested, &inventory, share)?;
    tracing::info!(
        "{}: macOS bandwidth choice asked={}x{}@{} chosen={}x{}@{} native={} estimated_raw_bytes_per_second={} share_bytes_per_second={} (fixed share of the unknown shared bus; dimensions/rate reduced only when inventory or budget requires it)",
        device.name,
        requested.width,
        requested.height,
        requested.fps,
        chosen.video.width,
        chosen.video.height,
        chosen.video.fps,
        String::from_utf8_lossy(&chosen.native.offered.mode.fourcc),
        chosen.bytes_per_second,
        share
    );
    device.video = Some(chosen.video);
    Ok((device, Some(share)))
}

/// Opens a camera on this OS and starts it streaming.
///
/// # Errors
///
/// The device's error.
pub fn open_camera(node: &str, device: &Device) -> std::io::Result<(Box<dyn Camera>, Opened)> {
    open_camera_with_share(node, device, None)
}

fn open_camera_with_share(
    node: &str,
    device: &Device,
    _mac_raw_share: Option<u64>,
) -> std::io::Result<(Box<dyn Camera>, Opened)> {
    let video = device.video().ok_or_else(|| std::io::Error::other("not a camera"))?;
    #[cfg(target_os = "linux")]
    let (mut source, opened) = {
        let (s, o) = crate::v4l2::V4l2Source::open(std::path::Path::new(node), &video)?;
        (Box::new(s) as Box<dyn Camera>, o)
    };
    #[cfg(target_os = "macos")]
    let (mut source, opened) = {
        let (s, o) = match _mac_raw_share {
            Some(share) => crate::avfoundation::AvfSource::open_with_share(node, &video, share)?,
            None => crate::avfoundation::AvfSource::open(node, &video)?,
        };
        (Box::new(s) as Box<dyn Camera>, o)
    };
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    let (mut source, opened): (Box<dyn Camera>, Opened) = {
        let _ = (node, &video);
        return Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "no camera source on this OS"));
    };
    // Open means streaming: the first frame is in, so the camera holds its bandwidth.
    source.start()?;
    let mut opened = opened;
    source.refine(&mut opened);
    Ok((source, opened))
}

/// Opens a microphone on this OS.
///
/// # Errors
///
/// The device's error.
pub fn open_microphone(node: &str) -> std::io::Result<(Box<dyn Microphone>, PcmFormat)> {
    #[cfg(target_os = "linux")]
    {
        let (s, f) = crate::pipewire::PipewireSource::open(node)?;
        Ok((Box::new(s), f))
    }
    #[cfg(target_os = "macos")]
    {
        let (s, f) = crate::coreaudio::CoreAudioSource::open(node)?;
        Ok((Box::new(s), f))
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = node;
        Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "no microphone source on this OS"))
    }
}

/// The bytes a camera claims its Tier 0 slot class by: its planned frame size (`continuum_core::camera_frame_bound`, the
/// size the broker reserved its slots at), or the driver's buffer if that's bigger. Claiming by the driver's buffer
/// alone picked a smaller class the broker never reserved for it whenever another camera added one (#2001).
#[must_use]
pub fn slot_need(max_frame_bytes: u32, width: u32, height: u32, shape: &str) -> u64 {
    let raw = cfg!(target_os = "macos") || shape == "UYVY";
    continuum_core::camera_frame_bound(width, height, raw).unwrap_or(0).max(u64::from(max_frame_bytes))
}

/// The slot class a frame of `need` bytes goes in: the smallest that holds it (the big ones stay free for big
/// payloads), as its index and slot size. `None` when none is big enough.
#[must_use]
pub fn pick_class(classes: &[continuum_client::SizeClass], need: u64) -> Option<(u16, u64)> {
    classes
        .iter()
        .enumerate()
        .filter(|(_, c)| c.slab_bytes >= need && c.count > 0)
        .min_by_key(|(_, c)| c.slab_bytes)
        .and_then(|(i, c)| Some((u16::try_from(i).ok()?, c.slab_bytes)))
}

/// Runs an opened camera: registers its stream, publishes into the broker's Tier 0 arena, and re-registers after a
/// wake.
/// The device is already open and streaming (the worker has said `opened`).
#[allow(clippy::too_many_arguments)]
pub fn camera(
    rt: &tokio::runtime::Handle,
    client: &mut Client,
    device: &Device,
    (node, socket, manifest): (&str, &std::path::Path, &std::path::Path),
    source: &mut dyn Camera,
    opened: &Opened,
    clocks: &dyn Clocks,
    stop: &AtomicBool,
) -> WorkerEnd {
    use crate::register::{encoded_video_spec, video_spec};
    use crate::video::{CameraConfig, CameraCounts, CameraExit, run_camera};

    let Some(mut video) = device.video() else { return WorkerEnd::BadSetup };
    // The stream registers what the camera delivers (on macOS, UYVY whatever the manifest asked).
    video.format.clone_from(&opened.shape);
    tracing::info!(
        "{}: {}x{} {} at {} ns a frame{}",
        device.name,
        opened.mode.width,
        opened.mode.height,
        opened.shape,
        opened.frame_interval_ns,
        opened.hold.map(|h| format!(", {h}")).unwrap_or_default()
    );
    // Frames go into the broker's arena (from Welcome), in the smallest slot class that fits one.
    let Some(mapped) = client.arenas().first() else {
        tracing::error!("{}: the broker described no Tier 0 arena", device.name);
        return WorkerEnd::Memory;
    };
    let need = slot_need(opened.max_frame_bytes, opened.mode.width, opened.mode.height, &opened.shape);
    let Some((class, slot_bytes)) = pick_class(&mapped.info.classes, need) else {
        tracing::error!(
            "{}: no Tier 0 slot class holds a {} byte frame (classes {:?})",
            device.name,
            opened.max_frame_bytes,
            mapped.info.classes
        );
        return WorkerEnd::Memory;
    };
    let arena = mapped.arena.clone();
    let slot_bytes = usize::try_from(slot_bytes).unwrap_or(usize::MAX);
    let stream_name = device.stream_name("video");
    let mut anchor = take_anchor(clocks);
    let spec = match encoded_video_spec(device, &video, &anchor, class) {
        Ok(Some(spec)) => spec,
        Ok(None) => video_spec(device, &video, &anchor, class),
        Err(error) => {
            tracing::error!("{}: {error}", device.name);
            return WorkerEnd::BadSetup;
        }
    };
    let encoding = video.encoding_for(device.kind).filter(|settings| settings.enabled);
    #[cfg(feature = "ffmpeg")]
    let pipeline = match &encoding {
        Some(settings) => {
            match crate::encoding::Pipeline::new(opened.mode.width, opened.mode.height, video.fps, settings.clone()) {
                Ok(pipeline) => Some(pipeline),
                Err(error) => {
                    tracing::error!("{}: cannot encode H.264: {error}", device.name);
                    return WorkerEnd::BadSetup;
                }
            }
        }
        None => None,
    };
    #[cfg(not(feature = "ffmpeg"))]
    if encoding.is_some() {
        tracing::error!("{}: H.264 capture requires rebuilding continuum-capture with the ffmpeg feature", device.name);
        return WorkerEnd::BadSetup;
    }
    let registered = match rt.block_on(client.register_stream(&spec)) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("{}: the broker refused {stream_name}: {e}", device.name);
            return WorkerEnd::Broker;
        }
    };
    let meta = StreamMeta {
        stream: StreamId(registered.stream_id),
        format: if encoding.is_some() || opened.shape == "MJPEG" {
            PayloadFormat::EncodedVideo
        } else {
            PayloadFormat::RawBytes
        },
        shape_name: if encoding.is_some() { crate::rtsp::SHAPE.into() } else { opened.shape.clone() },
        anchor,
        residual_offset_ns: device.residual_offset_ns.get("video").copied().unwrap_or(0),
    };
    // No handles ring until the broker carves rings (#634): live readers query the stream meanwhile.
    let stream = VideoStream::new(meta, arena, class, u64::from(std::process::id()), None);
    #[cfg(feature = "ffmpeg")]
    let mut stream: Box<dyn crate::video::CameraPublisher> = match pipeline {
        Some(pipeline) => Box::new(crate::video::EncodedPublisher { stream, pipeline, slot_bytes }),
        None => Box::new(stream),
    };
    #[cfg(not(feature = "ffmpeg"))]
    let mut stream: Box<dyn crate::video::CameraPublisher> = Box::new(stream);
    let cfg = CameraConfig {
        mode: opened.mode,
        stride: opened.stride,
        frame_interval_ns: opened.frame_interval_ns,
        flip: video.flip,
        stall_ns: video.stall_ms * 1_000_000,
        slot_bytes,
    };
    let mut counts = CameraCounts::default();
    // Auto exposure (#535): the camera's own thread, offered a frame a second.
    let mut steer = crate::steer::start(
        crate::steer::SteerSetup {
            camera: device.name.clone(),
            node: node.to_owned(),
            shape: opened.shape.clone(),
            width: opened.mode.width,
            height: opened.mode.height,
            stride: opened.stride,
            frame_interval_ns: opened.frame_interval_ns,
            settings: crate::exposure::Settings { target: video.exposure_target, ..Default::default() },
            socket: socket.to_owned(),
            manifest: manifest.to_owned(),
            video: video.clone(),
        },
        rt,
    );
    let mut offer = |at: u64, bytes: &[u8]| {
        if let Some(s) = steer.as_mut() {
            s.offer(at, bytes);
        }
    };
    // The counts every 10 s are DEBUG, and INFO once every few minutes (#59).
    let mut clock = continuum_log::Periodic::default();
    loop {
        let name = &device.name;
        let mut health = |c: &CameraCounts| continuum_log::periodic!(clock, "{name}: {c:?}");
        let exit =
            run_camera(source, stream.as_mut(), client, clocks, &cfg, stop, &mut counts, &mut health, &mut offer);
        tracing::info!("{}: {counts:?}", device.name);
        match exit {
            CameraExit::Woke(new) => {
                anchor = new;
                let mut refreshed = spec.clone();
                refreshed.anchor = Some(anchor.to_schema());
                if let Err(e) = rt.block_on(client.register_stream(&refreshed)) {
                    tracing::warn!("{}: re-registering after a wake failed: {e}", device.name);
                    return WorkerEnd::Broker;
                }
                stream.set_anchor(anchor);
            }
            CameraExit::Stopped => return WorkerEnd::Stopped,
            CameraExit::Stalled => return WorkerEnd::Stalled,
            CameraExit::Source(e) => {
                tracing::warn!("{}: {e}", device.name);
                return WorkerEnd::Device;
            }
            CameraExit::Publish(e) => {
                tracing::warn!("{}: {e}", device.name);
                return WorkerEnd::Broker;
            }
        }
    }
}

/// Runs an opened microphone: registers its stream, publishes into a standalone ring, and re-registers after a
/// wake. The device is already open (the worker has said `opened`).
#[allow(clippy::too_many_arguments)]
pub fn microphone(
    rt: &tokio::runtime::Handle,
    client: &mut Client,
    device: &Device,
    source: &mut dyn Microphone,
    pcm: PcmFormat,
    clocks: &dyn Clocks,
    stop: &AtomicBool,
) -> WorkerEnd {
    use crate::audio::{MicExit, run_microphone};
    use crate::reblock::Reblocker;
    use crate::register::audio_spec;

    let Some(audio) = device.audio() else { return WorkerEnd::BadSetup };
    tracing::info!("{}: {} Hz, {} channels, {}", device.name, pcm.rate_hz, pcm.channels, pcm.shape_name());
    let stream_name = device.stream_name("audio");
    // The broker carves the stream's ring at registration (stride, records, and generation in the lease), in its
    // Tier 0 arena; the worker writes it through the mapping Welcome gave.
    let mut anchor = take_anchor(clocks);
    let registered = match rt.block_on(client.register_stream(&audio_spec(device, &audio, pcm, &anchor))) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("{}: the broker refused {stream_name}: {e}", device.name);
            return WorkerEnd::Broker;
        }
    };
    let Some(lease) = registered.lease.clone() else {
        tracing::warn!("{}: the broker gave {stream_name} no ring", device.name);
        return WorkerEnd::Broker;
    };
    let (Some(carved), Some(arena)) = (lease.carved_ring(), client.arena(&lease.arena).cloned()) else {
        tracing::warn!("{}: {stream_name}'s lease names no ring in a mapped arena: {lease:?}", device.name);
        return WorkerEnd::Memory;
    };
    let meta = StreamMeta {
        stream: StreamId(registered.stream_id),
        format: PayloadFormat::RawBytes,
        shape_name: pcm.shape_name(),
        anchor,
        residual_offset_ns: device.residual_offset_ns.get("audio").copied().unwrap_or(0),
    };
    let ring = match arena.carved_ring_writer(&carved) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("{}: {e}", device.name);
            return WorkerEnd::Memory;
        }
    };
    let block_frames = pcm.frames_per_block(audio.block_ms);
    let mut stream = match AudioStream::new(meta, ring, pcm.rate_hz, block_frames, audio.batch_seconds) {
        Ok(s) => s.with_ring_locator(lease.arena.clone(), lease.arena_generation, lease.ring_index),
        Err(e) => {
            tracing::warn!("{}: {e}", device.name);
            return WorkerEnd::BadSetup;
        }
    };
    let frame_bytes = usize::from(pcm.channels) * usize::from(pcm.bytes_per_sample);
    let mut reblocker = Reblocker::new(frame_bytes, block_frames as usize, pcm.rate_hz);
    // A microphone that sends nothing for this long has stalled (the manifest's stall time is for cameras).
    let stall_ns = 5_000_000_000;
    // The counts every 10 s are DEBUG, and INFO once every few minutes (#59).
    let mut clock = continuum_log::Periodic::default();
    loop {
        let name = &device.name;
        let mut health = |s: &AudioStream| {
            let (batches, gaps) = s.counts();
            continuum_log::periodic!(clock, "{name}: {batches} batches, {gaps} gaps, drift {:?} ppm", s.drift_ppm());
        };
        let exit = run_microphone(source, &mut stream, &mut reblocker, client, clocks, stall_ns, stop, &mut health);
        let (batches, gaps) = stream.counts();
        tracing::info!(
            "{}: {batches} batches, {gaps} gaps, {} chunks dropped, drift {:?} ppm",
            device.name,
            source.dropped(),
            stream.drift_ppm()
        );
        match exit {
            MicExit::Woke(new) => {
                anchor = new;
                match rt.block_on(client.register_stream(&audio_spec(device, &audio, pcm, &anchor))) {
                    // The same broker run keeps the stream's ring; a new one (the broker restarted) means
                    // starting over on it, which the supervisor does.
                    Ok(again) if again.lease.as_ref().and_then(|l| l.carved_ring()) == Some(carved) => {}
                    Ok(_) => {
                        tracing::warn!("{}: the stream's ring moved; restarting on the new one", device.name);
                        return WorkerEnd::Broker;
                    }
                    Err(e) => {
                        tracing::warn!("{}: re-registering after a wake failed: {e}", device.name);
                        return WorkerEnd::Broker;
                    }
                }
            }
            MicExit::Stopped => return WorkerEnd::Stopped,
            MicExit::Stalled => return WorkerEnd::Stalled,
            MicExit::Source(e) => {
                tracing::warn!("{}: {e}", device.name);
                return WorkerEnd::Device;
            }
            MicExit::Publish(e) => {
                tracing::warn!("{}: {e}", device.name);
                return WorkerEnd::Broker;
            }
        }
    }
}

/// Set by `SIGTERM` or `SIGINT`: the worker finishes its batch and exits.
pub static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

/// Routes `SIGTERM` and `SIGINT` to [`STOP`].
pub fn install_stop_signals() {
    // SAFETY: the handler only stores to an atomic, which is async-signal-safe.
    unsafe {
        let handler = on_signal as extern "C" fn(libc::c_int) as libc::sighandler_t;
        libc::signal(libc::SIGTERM, handler);
        libc::signal(libc::SIGINT, handler);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_parse_and_missing_ones_are_named() {
        let args = |s: &str| s.split(' ').map(String::from).collect::<Vec<_>>();
        let a = parse_args(args("--manifest m.toml --device facecam --node /dev/video0 --socket /run/b.sock")).unwrap();
        assert_eq!((a.device.as_str(), a.node.as_str()), ("facecam", "/dev/video0"));
        assert_eq!(parse_args(args("--device facecam")).unwrap_err(), "--manifest is required");
        assert!(parse_args(args("--frobnicate 1")).unwrap_err().contains("frobnicate"));
        assert_eq!(parse_args(args("--node")).unwrap_err(), "--node needs a value");
        let fixed =
            parse_args(args("--manifest m --device camera --node camera --socket b --mac-raw-share 13333333")).unwrap();
        assert_eq!(fixed.mac_raw_share, Some(13_333_333));
        for value in ["0", "-1", "invalid"] {
            let flags = format!("--manifest m --device camera --node camera --socket b --mac-raw-share {value}");
            assert!(parse_args(args(&flags)).unwrap_err().contains("positive bytes per second"));
        }
    }

    #[test]
    #[cfg(not(target_os = "macos"))] // a Mac's frames are raw, planned and claimed at their own size
    fn a_camera_claims_the_class_its_planned_frame_size_was_reserved_in() {
        use continuum_client::SizeClass;
        // devbox's cam0 (#2001): 960x540 MJPEG whose driver buffer is 1,036,800 bytes, which an RTSP camera's 1 MiB
        // class would hold. Its planned size, 960 x 2 x 540 + 64 KiB, is 1,102,336: the 2 MiB class the broker
        // reserved its slots in. A buffer bigger than the plan still wins.
        assert_eq!(slot_need(1_036_800, 960, 540, "MJPEG"), 1_102_336);
        assert_eq!(slot_need(3_000_000, 960, 540, "MJPEG"), 3_000_000);
        let classes = [SizeClass { slab_bytes: 1 << 20, count: 245 }, SizeClass { slab_bytes: 2 << 20, count: 900 }];
        assert_eq!(pick_class(&classes, slot_need(1_036_800, 960, 540, "MJPEG")), Some((1, 2 << 20)));
        assert_eq!(pick_class(&classes, 1_036_800), Some((0, 1 << 20)), "by the driver's buffer alone, as before");
    }

    #[test]
    fn a_frame_goes_in_the_smallest_class_that_holds_it() {
        use continuum_client::SizeClass;
        let classes = [
            SizeClass { slab_bytes: 1 << 20, count: 64 },
            SizeClass { slab_bytes: 64 << 10, count: 256 },
            SizeClass { slab_bytes: 4 << 20, count: 0 },
            SizeClass { slab_bytes: 8 << 20, count: 8 },
        ];
        assert_eq!(pick_class(&classes, 40_000), Some((1, 64 << 10)));
        assert_eq!(pick_class(&classes, 64 << 10), Some((1, 64 << 10)), "an exact fit");
        assert_eq!(pick_class(&classes, 2 << 20), Some((3, 8 << 20)), "an empty class is skipped");
        assert_eq!(pick_class(&classes, 9 << 20), None);
    }
}
