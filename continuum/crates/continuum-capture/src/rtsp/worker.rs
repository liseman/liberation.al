//! Network packets into the existing Tier 0 video publisher; the stored picture data stays encoded.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use continuum_client::{Client, StreamSpec, StreamStorage};
use continuum_core::schema::continuum as fb;
use continuum_core::{PayloadFormat, StreamId};

use super::input::Reader;
use super::{NetworkClock, SHAPE};
use crate::clock::{Clocks, WakeWatch, take_anchor};
use crate::manifest::Device;
use crate::publish::{Capture, PublishError, StreamMeta, VideoStream};
use crate::worker::{WorkerEnd, pick_class};

/// Runs one configured RTSP camera. Each source reconnect uses the supervisor's existing bounded backoff.
/// The URL is opened inside the worker process, never passed as a process argument or included in diagnostics.
#[allow(clippy::too_many_arguments)]
pub fn run(
    rt: &tokio::runtime::Handle,
    client: &mut Client,
    device: &Device,
    clocks: &dyn Clocks,
    stop: &AtomicBool,
    opened: impl FnOnce(),
) -> WorkerEnd {
    let Some(settings) = device.rtsp.as_ref() else { return WorkerEnd::BadSetup };
    let Some(video) = device.video() else { return WorkerEnd::BadSetup };
    let source = match Reader::open(settings.clone(), &device.name) {
        Ok(source) => source,
        Err(error) => {
            tracing::warn!("{}: {error}", device.name);
            return WorkerEnd::Device;
        }
    };
    let until = Instant::now() + Duration::from_millis(settings.read_timeout_ms);
    let first = loop {
        if stop.load(Ordering::Relaxed) {
            return WorkerEnd::Stopped;
        }
        match source.next(Duration::from_millis(100)) {
            Ok(Some(picture)) => break picture,
            Ok(None) if Instant::now() < until => {}
            Ok(None) => {
                tracing::warn!("{}: RTSP did not provide a decodable IDR before its deadline", device.name);
                return WorkerEnd::Stalled;
            }
            Err(error) => {
                tracing::warn!("{}: {error}", device.name);
                return WorkerEnd::Device;
            }
        }
    };
    let Some(mapped) = client.arenas().first() else { return WorkerEnd::Memory };
    let Some((class, slot_bytes)) = pick_class(&mapped.info.classes, u64::from(settings.max_access_unit_bytes)) else {
        tracing::error!("{}: no Tier 0 size class holds its configured RTSP access-unit bound", device.name);
        return WorkerEnd::Memory;
    };
    let arena = mapped.arena.clone();
    let budget_arena = arena.clone();
    let classes = mapped.info.classes.clone();
    let anchor = take_anchor(clocks);
    let spec = StreamSpec {
        name: device.stream_name("video"),
        kind: fb::StreamKind::Sensor,
        format: fb::PayloadFormat::EncodedVideo,
        shape_name: SHAPE.into(),
        shape_version: 1,
        nominal_rate_hz: source.info.nominal_rate_hz,
        storage: StreamStorage::Slab { class },
        anchor: Some(anchor.to_schema()),
        layer: fb::DomainLayer::Layer1Raw,
    };
    let registered = match rt.block_on(client.register_stream(&spec)) {
        Ok(registered) => registered,
        Err(error) => {
            tracing::warn!("{}: RTSP stream registration failed: {error}", device.name);
            return WorkerEnd::Broker;
        }
    };
    let mut publisher = VideoStream::new(
        StreamMeta {
            stream: StreamId(registered.stream_id),
            format: PayloadFormat::EncodedVideo,
            shape_name: SHAPE.into(),
            anchor,
            residual_offset_ns: device.residual_offset_ns.get("video").copied().unwrap_or(0),
        },
        arena,
        class,
        u64::from(std::process::id()),
        None,
    );
    tracing::info!(
        "{}: H.264 RTSP {}x{} at {} Hz, recording encoded access units without transcoding",
        device.name,
        source.info.width,
        source.info.height,
        source.info.nominal_rate_hz
    );
    opened();
    // One connection is one capture session: a reconnect runs this again, with a new epoch.
    let epoch = crate::publish::new_capture_epoch();
    let mut clock = NetworkClock::default();
    let mut wake = WakeWatch::new(clocks);
    let mut pending = Some(first);
    let mut budget =
        super::SlotBudget::new(settings.ring_bytes).with_share(super::class_share(video.fps, video.ring_seconds));
    let mut frames = 0u64;
    let mut pressure_dropped = 0u64;
    // Pictures that found Tier 0 full and were published after waiting for the broker to free space.
    let mut full_waited = 0u64;
    let mut overwritten = 0u64;
    // Pictures passed over after a drop while waiting for the next IDR: a drop costs these too (#2350).
    let mut idr_skipped = 0u64;
    let mut wait_for_idr = false;
    let mut last_packet = clocks.monotonic_ns();
    let mut last_health = last_packet;
    // The health line every 10 s is DEBUG, and INFO once every few minutes (#59); continuum_status reads it.
    let mut health = continuum_log::Periodic::default();
    loop {
        if stop.load(Ordering::Relaxed) {
            return WorkerEnd::Stopped;
        }
        if wake.woke(clocks) {
            tracing::warn!("{}: reconnecting RTSP after host sleep", device.name);
            return WorkerEnd::Device;
        }
        let now = clocks.monotonic_ns();
        if now.saturating_sub(last_health) >= crate::video::HEALTH_NS {
            continuum_log::periodic!(
                health,
                "{}: RTSP frames={frames} reader_dropped={} pressure_dropped={pressure_dropped} unflushed_overwrites={overwritten} full_waited={full_waited} idr_skipped={idr_skipped} re_anchored={}",
                device.name,
                source.dropped.load(Ordering::Relaxed),
                clock.re_anchored()
            );
            last_health = now;
        }
        let picture =
            match pending.take().map_or_else(|| source.next(Duration::from_millis(100)), |picture| Ok(Some(picture))) {
                Ok(Some(picture)) => picture,
                Ok(None) if now.saturating_sub(last_packet) < video.stall_ms.saturating_mul(1_000_000) => continue,
                Ok(None) => return WorkerEnd::Stalled,
                Err(error) => {
                    tracing::warn!("{}: {error}", device.name);
                    return WorkerEnd::Device;
                }
            };
        last_packet = now;
        if picture.bytes.len() as u64 > slot_bytes {
            pressure_dropped += 1;
            wait_for_idr = true;
            continue;
        }
        if wait_for_idr && !picture.key {
            idr_skipped += 1;
            continue;
        }
        let mono_ns = match clock.observe(picture.pts_ns, picture.dts_ns, picture.arrival_ns) {
            Ok(mono_ns) => mono_ns,
            Err(error) => {
                tracing::warn!("{}: {error}", device.name);
                return WorkerEnd::Device;
            }
        };
        if let Some(note) = clock.take_note() {
            tracing::info!("{}: {note}", device.name);
        }
        let Some((picture_class, allocated)) = pick_class(&classes, picture.bytes.len() as u64) else {
            pressure_dropped += 1;
            wait_for_idr = true;
            continue;
        };
        let capture = Capture {
            mono_ns,
            device_ts: picture.pts_ns as u64,
            arrival_ns: picture.arrival_ns,
            device_seq: picture.sequence,
            stride: 0,
            width: source.info.width,
            height: source.info.height,
            flip: video.flip,
            sample_rate_hz: 0,
            // RTP carries no presentation duration; libavformat fills a packet's from a frame-rate guess, which is
            // not source evidence, so an RTSP picture stays an instant.
            duration_ns: 0,
            capture_epoch: epoch,
        };
        let (published, retries) = super::retry_full(
            || {
                budget.publish(
                    &budget_arena,
                    &mut publisher,
                    (picture_class, allocated),
                    capture,
                    client,
                    &picture.bytes,
                )
            },
            super::FULL_RETRIES,
            || std::thread::sleep(super::FULL_PAUSE),
        );
        match published {
            Ok(result) => {
                frames += 1;
                full_waited += u64::from(retries > 0);
                overwritten += result.unflushed_overwrites as u64;
                wait_for_idr = false;
            }
            Err(PublishError::Shm(continuum_shm::Error::Full)) => {
                pressure_dropped += 1;
                wait_for_idr = true;
            }
            Err(error) => {
                tracing::warn!("{}: RTSP publish failed: {error}", device.name);
                return WorkerEnd::Broker;
            }
        }
    }
}
