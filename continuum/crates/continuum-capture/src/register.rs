//! Registering capture streams with the broker (docs/CONTINUUM.md, "Registration"): a stream's `StreamSpec` from
//! the manifest and the device's actual format, with the clock anchor taken when capture starts.

use continuum_client::{StreamSpec, StreamStorage};
use continuum_core::schema::continuum as fb;

use crate::clock::Anchor;
use crate::manifest::{AudioSettings, Device, VideoSettings};

/// A camera stream: the device's native payload (MJPEG by default) in a slot ring of size class `class`.
#[must_use]
pub fn video_spec(device: &Device, video: &VideoSettings, anchor: &Anchor, class: u16) -> StreamSpec {
    let compressed = video.format == "MJPEG";
    StreamSpec {
        name: device.stream_name("video"),
        kind: fb::StreamKind::Sensor,
        format: if compressed { fb::PayloadFormat::EncodedVideo } else { fb::PayloadFormat::RawBytes },
        shape_name: video.format.clone(),
        shape_version: 1,
        nominal_rate_hz: video.fps as f32,
        storage: StreamStorage::Slab { class },
        anchor: Some(anchor.to_schema()),
        layer: fb::DomainLayer::Layer1Raw,
    }
}

/// The H.264 output on the canonical `video` stream of a camera that encodes ([`VideoSettings::encoding_for`]).
/// Runtime encoder fallback must also emit H.264. Returns `None` when it doesn't encode. This does not change
/// [`video_spec`].
///
/// # Errors
/// Refuses encoding settings that do not validate for the supplied device and native format.
pub fn encoded_video_spec(
    device: &Device,
    video: &VideoSettings,
    anchor: &Anchor,
    class: u16,
) -> Result<Option<StreamSpec>, &'static str> {
    let Some(encoding) = video.encoding_for(device.kind) else { return Ok(None) };
    if !encoding.enabled {
        return Ok(None);
    }
    encoding.check(device.kind, &video.format)?;
    let mut spec = video_spec(device, video, anchor, class);
    spec.format = fb::PayloadFormat::EncodedVideo;
    spec.shape_name = crate::rtsp::SHAPE.into();
    Ok(Some(spec))
}

/// The PCM layout a microphone opened with: what its ring's records are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PcmFormat {
    /// Frames per second.
    pub rate_hz: u32,
    /// Every channel the device has (none is dropped).
    pub channels: u16,
    /// Bytes per sample: 2 for S16, 3 for S24_3LE, 4 for S32 or F32.
    pub bytes_per_sample: u16,
    /// Floating point samples (F32), rather than integers.
    pub float: bool,
}

impl PcmFormat {
    /// `PCM_S16LE`, `PCM_S24LE3`, `PCM_S32LE`, or `PCM_F32LE`, with the channel count: the stream's shape.
    #[must_use]
    pub fn shape_name(&self) -> String {
        let kind = match (self.float, self.bytes_per_sample) {
            (true, _) => "F32LE",
            (false, 2) => "S16LE",
            (false, 3) => "S24LE3",
            (false, _) => "S32LE",
        };
        format!("PCM_{kind}x{}", self.channels)
    }

    /// Frames in a block of `block_ms`.
    #[must_use]
    pub fn frames_per_block(&self, block_ms: u32) -> u64 {
        u64::from(self.rate_hz) * u64::from(block_ms) / 1_000
    }

    /// Bytes of PCM in a block of `block_ms`.
    #[must_use]
    pub fn block_bytes(&self, block_ms: u32) -> u32 {
        let bytes = self.frames_per_block(block_ms) * u64::from(self.channels) * u64::from(self.bytes_per_sample);
        u32::try_from(bytes).unwrap_or(u32::MAX)
    }

    /// Bytes in a ring record: the stamp, then a block of `block_ms`.
    #[must_use]
    pub fn record_bytes(&self, block_ms: u32) -> u32 {
        self.block_bytes(block_ms).saturating_add(crate::publish::PCM_RECORD_HEADER as u32)
    }
}

/// A microphone stream: one ring record per block (its stamp, then every channel at the device's native rate and
/// format).
#[must_use]
pub fn audio_spec(device: &Device, audio: &AudioSettings, pcm: PcmFormat, anchor: &Anchor) -> StreamSpec {
    StreamSpec {
        name: device.stream_name("audio"),
        kind: fb::StreamKind::Sensor,
        format: fb::PayloadFormat::RawBytes,
        shape_name: pcm.shape_name(),
        shape_version: 1,
        nominal_rate_hz: 1_000.0 / audio.block_ms as f32,
        storage: StreamStorage::Ring { stride_bytes: pcm.record_bytes(audio.block_ms), seconds: audio.ring_seconds },
        anchor: Some(anchor.to_schema()),
        layer: fb::DomainLayer::Layer1Raw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoded_routing_is_explicit_and_uses_the_canonical_video_identity() {
        let manifest = crate::manifest::Manifest::parse(
            "[[device]]\nname='synthetic'\nkind='camera'\nmatch={name='Synthetic'}\nvideo={encoding={enabled=true}}\n",
        )
        .unwrap();
        let device = &manifest.devices[0];
        let mut video = device.video().unwrap();
        let anchor = Anchor { boot: continuum_core::BootId([1; 16]), mono_ns: 10, utc_ns: 20, bracket_ns: 0 };
        let native = video_spec(device, &video, &anchor, 2);
        assert_eq!((native.name.as_str(), native.shape_name.as_str()), ("synthetic.video", "MJPEG"));
        let encoded = encoded_video_spec(device, &video, &anchor, 2).unwrap().unwrap();
        assert_eq!((encoded.name.as_str(), encoded.shape_name.as_str()), ("synthetic.video", "H264AnnexB"));
        assert_eq!(encoded.format, fb::PayloadFormat::EncodedVideo);
        assert_eq!(encoded.nominal_rate_hz, native.nominal_rate_hz);
        assert!(matches!(encoded.storage, StreamStorage::Slab { class: 2 }));
        video.encoding.as_mut().unwrap().enabled = false;
        assert!(encoded_video_spec(device, &video, &anchor, 2).unwrap().is_none());
        video.encoding = None;
        assert_eq!(
            encoded_video_spec(device, &video, &anchor, 2).unwrap().is_some(),
            cfg!(feature = "ffmpeg"),
            "an MJPEG camera with no encoding settings encodes when the build can"
        );
        video.encoding = Some(crate::manifest::VideoEncodingSettings::default());
        video.encoding.as_mut().unwrap().enabled = true;
        video.format = "UYVY".into();
        assert!(encoded_video_spec(device, &video, &anchor, 2).is_err());
        video.encoding.as_mut().unwrap().enabled = false;
        assert!(
            encoded_video_spec(device, &video, &anchor, 2).unwrap().is_none(),
            "disabled encoding preserves negotiated native formats"
        );
    }

    #[test]
    fn the_wave3s_blocks_are_ring_records() {
        let wave = PcmFormat { rate_hz: 48_000, channels: 1, bytes_per_sample: 3, float: false };
        assert_eq!(wave.shape_name(), "PCM_S24LE3x1");
        assert_eq!(wave.frames_per_block(10), 480);
        assert_eq!(wave.block_bytes(10), 1_440);
        assert_eq!(wave.record_bytes(10), 1_456, "with the 16-byte stamp");
        let xvf = PcmFormat { rate_hz: 16_000, channels: 6, bytes_per_sample: 2, float: false };
        assert_eq!((xvf.shape_name(), xvf.block_bytes(20)), ("PCM_S16LEx6".into(), 3_840));
    }
}
