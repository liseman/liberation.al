//! The CoreAudio microphone source (macOS), through cpal: an input stream on one device at its default (native)
//! rate and every channel, as f32. Each buffer's capture time is its `mHostTime` (CoreAudio's host time for the
//! buffer's first frame, on `mach_absolute_time`, the clock capture stamps with on macOS). cpal also offers a
//! "capture" time with its own latency estimate taken off; which of the two is when the first frame left the
//! microphone is settled on the Mac run (the worker logs both for the first buffer). Buffers go to the worker
//! through a [`ChunkSender`], so the worker's loop is the same as for PipeWire.

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::audio::{ChannelSource, Chunk, ChunkSource, chunk_channel};
use crate::manifest::DeviceKind;
use crate::register::PcmFormat;
use crate::supervisor::DeviceInfo;

/// How many chunks may wait for the worker before the callback drops them (counted, and seen as a gap).
const CHUNK_DEPTH: usize = 64;

/// The key prefix for a CoreAudio device; the rest is cpal's stable device id.
pub const KEY_PREFIX: &str = "coreaudio:";

/// Every input device CoreAudio has now, as hotplug devices: keyed and opened by cpal's id, named by its
/// description.
#[must_use]
pub fn input_devices() -> Vec<DeviceInfo> {
    let host = cpal::default_host();
    let Ok(devices) = host.input_devices() else { return Vec::new() };
    devices
        .filter_map(|d| {
            let id = d.id().ok()?.to_string();
            let name = d.description().ok()?.name().to_owned();
            Some(DeviceInfo {
                key: format!("{KEY_PREFIX}{id}"),
                kind: DeviceKind::Microphone,
                name,
                usb: None,
                node: id.clone(),
                paths: vec![id],
            })
        })
        .collect()
}

/// A CoreAudio input stream, running.
pub struct CoreAudioSource {
    chunks: ChannelSource,
    // Taken on drop, so the audio unit is stopped and then released as separate, timed steps (#1388).
    stream: Option<cpal::Stream>,
}

impl Drop for CoreAudioSource {
    fn drop(&mut self) {
        use crate::teardown::step;
        let Some(stream) = self.stream.take() else { return };
        // pause is AudioOutputUnitStop, which returns once the unit has stopped calling back; the drop then
        // uninitializes and disposes of the unit and removes cpal's device listeners.
        if let Err(e) = step("stop audio unit", || stream.pause()) {
            tracing::warn!("coreaudio stop: {e}");
        }
        step("release audio unit", || drop(stream));
    }
}

impl CoreAudioSource {
    /// Opens the input device whose cpal id is `node`, at its default rate and channel count, as f32.
    ///
    /// # Errors
    ///
    /// `NotFound` if no input device has that id, or cpal's error.
    pub fn open(node: &str) -> io::Result<(Self, PcmFormat)> {
        let host = cpal::default_host();
        let device = host
            .input_devices()
            .map_err(io::Error::other)?
            .find(|d| d.id().is_ok_and(|id| id.to_string() == node))
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("no input device {node}")))?;
        let default = device.default_input_config().map_err(io::Error::other)?;
        let config = cpal::StreamConfig {
            channels: default.channels(),
            sample_rate: default.sample_rate(),
            buffer_size: cpal::BufferSize::Default,
        };
        let format =
            PcmFormat { rate_hz: config.sample_rate, channels: config.channels, bytes_per_sample: 4, float: true };
        let (tx, chunks) = chunk_channel(CHUNK_DEPTH);
        let failed = tx.clone();
        let reported = Arc::new(AtomicBool::new(false));
        let stream = device
            .build_input_stream(
                config,
                move |data: &[f32], info: &cpal::InputCallbackInfo| {
                    let t = info.timestamp();
                    let host_ns = u64::try_from(t.callback.as_nanos()).unwrap_or(u64::MAX);
                    if !reported.swap(true, Ordering::Relaxed) {
                        tracing::info!(
                            "coreaudio time: host {host_ns} ns, cpal capture {} ns, {} samples",
                            t.capture.as_nanos(),
                            data.len()
                        );
                    }
                    let bytes: Vec<u8> = data.iter().flat_map(|s| s.to_le_bytes()).collect();
                    tx.send(Chunk { bytes, captured_ns: host_ns });
                },
                move |e| failed.fail(format!("coreaudio: {e}")),
                None,
            )
            .map_err(io::Error::other)?;
        stream.play().map_err(io::Error::other)?;
        Ok((Self { chunks, stream: Some(stream) }, format))
    }

    /// Chunks dropped because the worker fell behind.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.chunks.dropped()
    }
}

impl crate::audio::Microphone for CoreAudioSource {
    fn dropped(&self) -> u64 {
        CoreAudioSource::dropped(self)
    }
}

impl ChunkSource for CoreAudioSource {
    fn next_chunk(&mut self, timeout_ms: u32) -> io::Result<Option<Chunk>> {
        self.chunks.next_chunk(timeout_ms)
    }
}
