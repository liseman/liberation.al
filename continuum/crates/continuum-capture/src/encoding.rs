//! Canonical MJPEG-camera H.264 adapter. The hardware encoder is NVENC, or VideoToolbox on macOS; its
//! failure, at open or at run time, selects CPU H.264 (libx264, or OpenH264 where FFmpeg has no libx264), never raw
//! JPEG.
//! Each JPEG is decoded on the CPU straight to NV12, its own Y, Cb, and Cr repacked with no trip through RGB
//! (`continuum_tier::JpegNv12`), into buffers kept from frame to frame.
//! Source capture headers remain the publisher's responsibility; every accepted image yields exactly
//! one access unit with its original monotonic timestamp and attested duration.

use continuum_tier::h264::{H264Encoder, H264Settings, SOFTWARE_H264_ENCODERS};
use continuum_tier::{JpegNv12, TierError};

use crate::manifest::VideoEncodingSettings;
use crate::publish::Capture;

const MAX_PIXELS: u64 = 16_777_216;
/// The hardware encoders capture uses; a run-time failure of either falls back to a software encoder.
const HARDWARE: [&str; 2] = ["h264_nvenc", "h264_videotoolbox"];

/// This platform's hardware H.264 encoder: VideoToolbox on macOS (the Mac), NVENC elsewhere.
#[must_use]
pub const fn hardware_encoder() -> &'static str {
    if cfg!(target_os = "macos") { "h264_videotoolbox" } else { "h264_nvenc" }
}

/// Opens `name`, a hardware encoder: NVENC on the configured GPU at constant quality, or VideoToolbox in its
/// low-delay realtime mode at the settings' bit rate.
fn open_hardware(
    name: &str,
    width: u32,
    height: u32,
    settings: H264Settings,
    config: &VideoEncodingSettings,
) -> Result<H264Encoder, TierError> {
    if name == "h264_nvenc" {
        H264Encoder::open_nvenc(width, height, settings, config.gpu, config.cq)
    } else {
        H264Encoder::open_with(&[name], width, height, settings)
    }
}
const MAX_JPEG_BYTES: usize = 16 * 1024 * 1024;

fn invalid(message: &str) -> TierError {
    TierError::Codec(format!("capture H.264: {message}"))
}

/// Stateful, immediate-output encoding for one fixed camera layout.
pub struct Pipeline {
    width: u32,
    height: u32,
    settings: H264Settings,
    interval_ns: u64,
    encoder: H264Encoder,
    previous: Option<(u64, u64, u32, u64)>,
    /// The JPEG decoder's working buffer and the NV12 picture it fills, both reused from frame to frame.
    decoder: JpegNv12,
    nv12: Vec<u8>,
}

impl Pipeline {
    /// Opens this platform's hardware encoder ([`hardware_encoder`]), falling back loudly to a low-delay software
    /// encoder on failure.
    /// # Errors
    /// Invalid layout/settings or both hardware and software encoder initialization failing.
    pub fn new(width: u32, height: u32, fps: u32, config: VideoEncodingSettings) -> Result<Self, TierError> {
        if width == 0
            || height == 0
            || !width.is_multiple_of(2)
            || !height.is_multiple_of(2)
            || u64::from(width) * u64::from(height) > MAX_PIXELS
        {
            return Err(invalid("dimensions must be positive, even, and at most 16 million pixels"));
        }
        if !(1..=240).contains(&fps)
            || config.gpu > i32::MAX as u32
            || config.cq > 51
            || !(100..=10_000).contains(&config.keyframe_interval_ms)
        {
            return Err(invalid("invalid FPS, GPU ordinal, quality, or keyframe interval"));
        }
        let gop_frames = fps
            .checked_mul(config.keyframe_interval_ms)
            .ok_or_else(|| invalid("GOP interval overflow"))?
            .div_ceil(1000)
            .max(1);
        let settings = H264Settings {
            gop_frames,
            frame_rate: fps,
            bit_rate: continuum_tier::raw::bit_rate(width, height, fps),
            full_range: true,
            low_delay: true,
            threads: 1,
        };
        let hardware = hardware_encoder();
        let encoder = match open_hardware(hardware, width, height, settings, &config) {
            Ok(encoder) => encoder,
            Err(error) => {
                tracing::warn!(
                    "{hardware} initialization failed ({error}); using CPU H.264, increased CPU load expected"
                );
                H264Encoder::open_with(SOFTWARE_H264_ENCODERS, width, height, settings)?
            }
        };
        Ok(Self {
            width,
            height,
            settings,
            interval_ns: u64::from(config.keyframe_interval_ms) * 1_000_000,
            encoder,
            previous: None,
            decoder: JpegNv12::default(),
            nv12: Vec::new(),
        })
    }

    /// Actual backend, including runtime software fallback, for caller health reporting.
    #[must_use]
    pub fn backend_name(&self) -> &str {
        self.encoder.encoder_name()
    }

    /// A frame was not published: the next published frame must not depend on its references.
    pub fn break_chain(&mut self) {
        self.previous = None;
    }

    /// Decode one JPEG straight to NV12 (its own Y, Cb, and Cr, never through RGB) and encode it, without altering
    /// any capture metadata.
    /// # Errors
    /// Invalid JPEG/layout/time, encoder failure, or output not exactly one matching access unit.
    pub fn encode(&mut self, capture: Capture, jpeg: &[u8]) -> Result<Vec<u8>, TierError> {
        // The picture buffer is lent to this frame's work and kept for the next, whatever becomes of the frame.
        let mut nv12 = std::mem::take(&mut self.nv12);
        let encoded = self.encode_into(capture, jpeg, &mut nv12);
        self.nv12 = nv12;
        encoded
    }

    fn encode_into(&mut self, capture: Capture, jpeg: &[u8], nv12: &mut Vec<u8>) -> Result<Vec<u8>, TierError> {
        if (capture.width, capture.height) != (self.width, self.height) {
            return Err(invalid("capture dimensions changed; reopen the encoder for the new layout"));
        }
        if jpeg.is_empty() || jpeg.len() > MAX_JPEG_BYTES {
            return Err(invalid("JPEG must contain at most 16 MiB"));
        }
        if jpeg_dimensions(jpeg)? != (self.width, self.height) {
            return Err(invalid("JPEG dimensions do not match the configured capture layout"));
        }
        if self.decoder.decode(jpeg, nv12)? != (self.width, self.height) {
            return Err(invalid("decoded JPEG layout changed"));
        }
        let end =
            capture.mono_ns.checked_add(capture.duration_ns).ok_or_else(|| invalid("capture duration overflow"))?;
        let bucket = capture.mono_ns / self.interval_ns;
        let key = requires_keyframe(self.previous, capture, self.interval_ns);
        let encoded = self.encode_one(nv12, capture.mono_ns, end, key);
        let bytes = match encoded {
            Ok(bytes) => bytes,
            // A hardware encoder that fails, or breaks the one-access-unit-per-frame contract (an encoder that holds a
            // frame back), is replaced for good: the frame is encoded again on the CPU as a keyframe.
            Err(error) if HARDWARE.contains(&self.encoder.encoder_name()) => {
                tracing::warn!(
                    "{} runtime failure ({error}); switching to CPU H.264, increased CPU load expected",
                    self.encoder.encoder_name()
                );
                self.break_chain();
                self.encoder = H264Encoder::open_with(SOFTWARE_H264_ENCODERS, self.width, self.height, self.settings)?;
                self.encode_one(nv12, capture.mono_ns, end, true)?
            }
            Err(error) => {
                self.break_chain();
                return Err(error);
            }
        };
        self.previous = Some((capture.mono_ns, capture.device_seq, capture.capture_epoch, bucket));
        Ok(bytes)
    }

    fn encode_one(&mut self, nv12: &[u8], start: u64, end: u64, key: bool) -> Result<Vec<u8>, TierError> {
        let mut units = self.encoder.encode_keyframe(nv12, start, end, key)?;
        if units.len() != 1 {
            return Err(invalid("low-delay encoder did not return exactly one immediate access unit"));
        }
        let unit = units.pop().ok_or_else(|| invalid("encoder returned no access unit"))?;
        if unit.start_ns != start || unit.end_ns != end || unit.bytes.is_empty() {
            return Err(invalid("encoder output does not preserve the current capture identity and duration"));
        }
        if key && (!unit.key || !continuum_tier::gop::h264_idr(&unit.bytes)) {
            return Err(invalid("encoder did not honor the independently decodable frame request"));
        }
        Ok(unit.bytes)
    }
}

/// Read SOF dimensions before the JPEG decoder can allocate its picture. Entropy data is never scanned.
fn jpeg_dimensions(bytes: &[u8]) -> Result<(u32, u32), TierError> {
    if !bytes.starts_with(&[0xff, 0xd8]) {
        return Err(invalid("input is not a JPEG"));
    }
    let mut at = 2;
    while at < bytes.len() {
        if bytes[at] != 0xff {
            return Err(invalid("invalid JPEG marker"));
        }
        while at < bytes.len() && bytes[at] == 0xff {
            at += 1;
        }
        let marker = *bytes.get(at).ok_or_else(|| invalid("truncated JPEG marker"))?;
        at += 1;
        if marker == 0xda || marker == 0xd9 {
            break;
        }
        if marker == 0x01 || (0xd0..=0xd7).contains(&marker) {
            continue;
        }
        let size = bytes.get(at..at + 2).ok_or_else(|| invalid("truncated JPEG segment"))?;
        let length = usize::from(u16::from_be_bytes([size[0], size[1]]));
        if length < 2 {
            return Err(invalid("invalid JPEG segment length"));
        }
        let segment = bytes.get(at..at + length).ok_or_else(|| invalid("truncated JPEG segment"))?;
        if matches!(marker,0xc0..=0xc3|0xc5..=0xc7|0xc9..=0xcb|0xcd..=0xcf) {
            if segment.len() < 8 {
                return Err(invalid("truncated JPEG dimensions"));
            }
            let height = u32::from(u16::from_be_bytes([segment[3], segment[4]]));
            let width = u32::from(u16::from_be_bytes([segment[5], segment[6]]));
            return Ok((width, height));
        }
        at += length;
    }
    Err(invalid("JPEG has no frame dimensions before entropy data"))
}

fn requires_keyframe(previous: Option<(u64, u64, u32, u64)>, capture: Capture, interval_ns: u64) -> bool {
    previous.is_none_or(|(mono, seq, epoch, cut)| {
        capture.mono_ns <= mono
            || seq.checked_add(1) != Some(capture.device_seq)
            || capture.capture_epoch != epoch
            || capture.mono_ns / interval_ns != cut
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyframe_cuts_follow_capture_time_epoch_sequence_and_explicit_chain_breaks() {
        let previous = Some((1_900_000_000, 10, 1, 0));
        let next = Capture { mono_ns: 1_950_000_000, device_seq: 11, capture_epoch: 1, ..Capture::default() };
        assert!(!requires_keyframe(previous, next, 2_000_000_000));
        assert!(requires_keyframe(None, next, 2_000_000_000));
        assert!(requires_keyframe(previous, Capture { mono_ns: 2_000_000_000, ..next }, 2_000_000_000));
        assert!(requires_keyframe(previous, Capture { device_seq: 12, ..next }, 2_000_000_000));
        assert!(requires_keyframe(previous, Capture { capture_epoch: 2, ..next }, 2_000_000_000));
        assert!(requires_keyframe(previous, Capture { mono_ns: 1_800_000_000, ..next }, 2_000_000_000));
    }

    #[test]
    fn jpeg_layout_probe_checks_segment_lengths_before_reading_dimensions() {
        let header = [0xff, 0xd8, 0xff, 0xe0, 0, 4, 0, 0, 0xff, 0xc0, 0, 8, 8, 0, 48, 0, 64, 0];
        assert_eq!(jpeg_dimensions(&header).unwrap(), (64, 48));
        assert!(jpeg_dimensions(&header[..16]).is_err());
        assert!(jpeg_dimensions(&[0xff, 0xd8, 0xff, 0xe0, 0, 1]).is_err());
        assert!(jpeg_dimensions(&[0xff, 0xd8, 0xff, 0xda, 0, 8]).is_err());
        assert!(jpeg_dimensions(b"not JPEG").is_err());
    }

    fn cpu_pipeline() -> Pipeline {
        let settings = H264Settings {
            gop_frames: 60,
            frame_rate: 30,
            bit_rate: 300_000,
            full_range: true,
            low_delay: true,
            threads: 1,
        };
        Pipeline {
            width: 64,
            height: 48,
            settings,
            interval_ns: 2_000_000_000,
            encoder: H264Encoder::open_with(SOFTWARE_H264_ENCODERS, 64, 48, settings).unwrap(),
            previous: None,
            decoder: JpegNv12::default(),
            nv12: Vec::new(),
        }
    }

    fn jpeg(width: u16, height: u16) -> Vec<u8> {
        let mut bytes = Vec::new();
        jpeg_encoder::Encoder::new(&mut bytes, 90)
            .encode(
                &vec![120; usize::from(width) * usize::from(height) * 3],
                width,
                height,
                jpeg_encoder::ColorType::Rgb,
            )
            .unwrap();
        bytes
    }

    fn capture(mono_ns: u64, device_seq: u64, capture_epoch: u32) -> Capture {
        Capture {
            mono_ns,
            device_seq,
            capture_epoch,
            width: 64,
            height: 48,
            flip: true,
            device_ts: 123_456,
            arrival_ns: mono_ns + 500,
            stride: 128,
            duration_ns: 12_345,
            sample_rate_hz: 0,
        }
    }

    #[test]
    fn real_cpu_jpegs_emit_one_immediate_au_and_force_idr_at_every_reference_break() {
        let mut pipeline = cpu_pipeline();
        let picture = jpeg(64, 48);
        let mut decoder = continuum_tier::video::H264StreamDecoder::new().unwrap();
        let samples = [
            (capture(1_900_000_000, 10, 1), true),
            (capture(1_950_000_000, 11, 1), false),
            (capture(2_000_000_000, 12, 1), true),
            (capture(2_050_000_000, 14, 1), true),
            (capture(2_100_000_000, 15, 2), true),
        ];
        for (sample, key) in samples {
            let bytes = pipeline.encode(sample, &picture).unwrap();
            assert_eq!(continuum_tier::gop::h264_idr(&bytes), key);
            assert!(!bytes.starts_with(&[0xff, 0xd8]), "published bytes must be H264, not the source JPEG");
            let pictures = decoder.push(&bytes, sample.device_seq).unwrap();
            assert_eq!(pictures.len(), 1, "one immediate picture per source AU");
            assert_eq!((pictures[0].tag, pictures[0].width, pictures[0].height), (sample.device_seq, 64, 48));
            assert_eq!(
                pipeline.previous,
                Some((sample.mono_ns, sample.device_seq, sample.capture_epoch, sample.mono_ns / 2_000_000_000))
            );
        }
        pipeline.break_chain();
        let instant = Capture { duration_ns: 0, ..capture(2_150_000_000, 16, 2) };
        let bytes = pipeline.encode(instant, &picture).unwrap();
        assert!(continuum_tier::gop::h264_idr(&bytes));
        assert_eq!(decoder.push(&bytes, 16).unwrap().len(), 1);
    }

    #[test]
    fn the_hardware_encoder_is_videotoolbox_on_a_mac_and_nvenc_elsewhere() {
        let want = if cfg!(target_os = "macos") { "h264_videotoolbox" } else { "h264_nvenc" };
        assert_eq!(hardware_encoder(), want);
        assert!(HARDWARE.contains(&hardware_encoder()), "its run-time failures fall back too");
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn a_hardware_encoder_that_wont_open_falls_back_to_software() {
        // No GPU 63 here (and NVENC refuses 64x48 anyway): the pipeline still opens, on the CPU.
        let config = VideoEncodingSettings { enabled: true, gpu: 63, cq: 36, ..Default::default() };
        let mut pipeline = Pipeline::new(64, 48, 30, config).unwrap();
        assert!(SOFTWARE_H264_ENCODERS.contains(&pipeline.backend_name()));
        let bytes = pipeline.encode(capture(1_900_000_000, 10, 1), &jpeg(64, 48)).unwrap();
        assert!(continuum_tier::gop::h264_idr(&bytes));
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn real_videotoolbox_jpegs_emit_one_immediate_au_and_force_idr_at_every_reference_break() {
        // On the Mac: VideoToolbox must open, give one decodable access unit per frame, honor every
        // keyframe request, and never fall back to the CPU.
        const W: u16 = 256;
        const H: u16 = 144;
        let config = VideoEncodingSettings { enabled: true, ..Default::default() };
        let mut pipeline = Pipeline::new(u32::from(W), u32::from(H), 30, config).unwrap();
        assert_eq!(
            pipeline.backend_name(),
            "h264_videotoolbox",
            "VideoToolbox didn't open; capture would encode on the CPU"
        );
        let picture = jpeg(W, H);
        let mut decoder = continuum_tier::video::H264StreamDecoder::new().unwrap();
        let samples = [
            (capture(1_900_000_000, 10, 1), true),
            (capture(1_950_000_000, 11, 1), false),
            (capture(2_000_000_000, 12, 1), true),
            (capture(2_050_000_000, 14, 1), true),
            (capture(2_100_000_000, 15, 2), true),
        ];
        for (sample, key) in samples {
            let sample = Capture { width: u32::from(W), height: u32::from(H), ..sample };
            let bytes = pipeline.encode(sample, &picture).unwrap();
            assert_eq!(pipeline.backend_name(), "h264_videotoolbox", "VideoToolbox failed at run time and fell back");
            if key {
                assert!(continuum_tier::gop::h264_idr(&bytes), "a cut must be an IDR");
            }
            let pictures = decoder.push(&bytes, sample.device_seq).unwrap();
            assert_eq!(pictures.len(), 1, "one immediate picture per source frame");
            assert_eq!((pictures[0].width, pictures[0].height), (u32::from(W), u32::from(H)));
        }
    }

    #[test]
    fn malformed_or_mismatched_jpegs_never_advance_the_real_encoder_reference_chain() {
        let mut pipeline = cpu_pipeline();
        let picture = jpeg(64, 48);
        let first = capture(1_000_000_000, 10, 1);
        pipeline.encode(first, &picture).unwrap();
        let previous = pipeline.previous;
        let next = capture(1_030_000_000, 11, 1);
        assert!(pipeline.encode(next, b"not JPEG").is_err());
        assert_eq!(pipeline.previous, previous);
        assert!(pipeline.encode(next, &jpeg(32, 48)).is_err());
        assert_eq!(pipeline.previous, previous);
        assert!(pipeline.encode(Capture { width: 66, ..next }, &picture).is_err());
        assert_eq!(pipeline.previous, previous);
        let bad_entropy = [0xff, 0xd8, 0xff, 0xc0, 0, 8, 8, 0, 48, 0, 64, 0];
        assert!(pipeline.encode(next, &bad_entropy).is_err());
        assert_eq!(pipeline.previous, previous);
        let bytes = pipeline.encode(next, &picture).unwrap();
        assert!(!continuum_tier::gop::h264_idr(&bytes), "rejected inputs must not reach or restart the encoder");
    }

    struct Named(continuum_shm::Arena);
    impl Drop for Named {
        fn drop(&mut self) {
            let _ = continuum_shm::Arena::unlink(self.0.name());
        }
    }
    fn arena() -> Named {
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let name =
            format!("/cnt.enc.{}.{}", std::process::id(), NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
        Named(
            continuum_shm::Arena::create(
                &name,
                &continuum_shm::ArenaConfig {
                    classes: vec![continuum_shm::SlabClass { slot_size: 64 * 1024, slot_count: 1 }],
                    ..Default::default()
                },
            )
            .unwrap(),
        )
    }
    #[derive(Default)]
    struct Notices(Vec<(continuum_core::HeaderBuf, Option<continuum_core::StorageLocator>)>);
    impl crate::publish::Sink for Notices {
        fn ingest(
            &mut self,
            header: continuum_core::HeaderBuf,
            location: Option<continuum_core::StorageLocator>,
        ) -> Result<(), crate::publish::SinkError> {
            self.0.push((header, location));
            Ok(())
        }
    }
    fn published_bytes(arena: &continuum_shm::Arena, published: crate::publish::Published) -> Vec<u8> {
        let lease = arena.lease(published.slot, std::time::Duration::from_secs(1)).unwrap();
        // SAFETY: this lease pins exactly the published source generation while its bytes are copied.
        unsafe { std::slice::from_raw_parts(lease.as_ptr(), lease.len()) }.to_vec()
    }

    #[test]
    fn real_encoded_publisher_preserves_capture_header_and_never_overwrites_unflushed_pressure() {
        use crate::video::CameraPublisher;
        use continuum_core::{BootId, PayloadFormat, StorageLocator, StreamId};
        let arena = arena();
        let anchor = crate::clock::Anchor {
            boot: BootId([1; 16]),
            mono_ns: 1_000_000_000,
            utc_ns: 1_800_000_000_000_000_000,
            bracket_ns: 7,
        };
        let meta = crate::publish::StreamMeta {
            stream: StreamId(7),
            format: PayloadFormat::EncodedVideo,
            shape_name: "H264AnnexB".into(),
            anchor,
            residual_offset_ns: 17_000_000,
        };
        let mut publisher = crate::video::EncodedPublisher {
            stream: crate::publish::VideoStream::new(meta, arena.0.clone(), 0, 42, None),
            pipeline: cpu_pipeline(),
            slot_bytes: 64 * 1024,
        };
        let mut notices = Notices::default();
        let picture = jpeg(64, 48);
        let first = capture(1_900_000_000, u64::from(u32::MAX) + 4, 23);
        let published = publisher.publish_frame(first, &picture, &mut notices).unwrap();
        let bytes = published_bytes(&arena.0, published);
        assert!(continuum_tier::gop::h264_idr(&bytes));
        let header = &notices.0[0].0;
        let native = header.header();
        let cap = native.capture().unwrap();
        assert_eq!(header.stream(), StreamId(7));
        assert_ne!(header.id().0, [0; 16]);
        assert_eq!(header.shape_name(), "H264AnnexB");
        assert_eq!(native.format(), PayloadFormat::EncodedVideo);
        let start = 1_800_000_000_883_000_000;
        assert_eq!((header.start_ns(), native.end_ns()), (start, start + 12_345));
        assert_eq!(native.seq(), first.device_seq as u32);
        assert_eq!(
            (cap.capture_mono_ns(), cap.device_ts(), cap.arrival_ns(), cap.device_seq()),
            (first.mono_ns, first.device_ts, first.arrival_ns, first.device_seq)
        );
        assert_eq!((cap.width(), cap.height(), cap.stride(), cap.flip(), cap.capture_epoch()), (64, 48, 0, true, 23));
        let Some(StorageLocator::Shm { length, slot, slot_generation, .. }) = &notices.0[0].1 else {
            panic!("encoded frame must use the arena");
        };
        assert_eq!(
            (*length, *slot, *slot_generation),
            (bytes.len() as u64, published.slot.index, published.slot.generation)
        );
        let second = capture(1_930_000_000, first.device_seq + 1, 23);
        assert!(matches!(
            publisher.publish_frame(second, &picture, &mut notices),
            Err(crate::publish::PublishError::Shm(continuum_shm::Error::Full))
        ));
        assert_eq!(notices.0.len(), 1, "an unpublished frame must have no source notice");
        assert_eq!(
            published_bytes(&arena.0, published),
            bytes,
            "unflushed bytes must not be overwritten even without a pin"
        );
        // Flushed, and nobody reclaims it: the publisher recycles the slot itself rather than waiting for the
        // broker's sweep.
        arena.0.mark_flushed(published.slot).unwrap();
        let third = Capture { duration_ns: 0, ..capture(1_960_000_000, first.device_seq + 2, 23) };
        let next = publisher.publish_frame(third, &picture, &mut notices).unwrap();
        let independent = published_bytes(&arena.0, next);
        assert!(
            continuum_tier::gop::h264_idr(&independent),
            "pressure breaks the next published frame's reference chain"
        );
        let mut decoder = continuum_tier::video::H264StreamDecoder::new().unwrap();
        let decoded = decoder.push(&independent, third.device_seq).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].tag, third.device_seq);
        assert_eq!(
            (notices.0[1].0.start_ns(), notices.0[1].0.header().end_ns()),
            (start + 60_000_000, start + 60_000_000)
        );
        assert_eq!(next.unflushed_overwrites, 0);
    }

    #[test]
    #[ignore = "requires explicit NVENC GPU validation"]
    fn real_nvenc_jpegs_preserve_source_headers_and_independent_cuts() {
        use crate::video::CameraPublisher;
        use continuum_core::{BootId, PayloadFormat, StreamId};
        let gpu = std::env::var("CONTINUUM_TEST_CUDA_GPU")
            .map_or(Ok(0), |value| value.parse::<u32>())
            .expect("CONTINUUM_TEST_CUDA_GPU must be a CUDA ordinal");
        let config = VideoEncodingSettings { enabled: true, gpu, cq: 36, ..Default::default() };
        assert_eq!(config.keyframe_interval_ms, 2000);
        // NVENC refuses frames this small: 64x48 is under its minimum, so this test uses a size it encodes.
        const W: u16 = 256;
        const H: u16 = 144;
        let pipeline = Pipeline::new(u32::from(W), u32::from(H), 30, config).unwrap();
        assert_eq!(pipeline.backend_name(), "h264_nvenc", "CPU fallback cannot validate NVENC");
        let arena = arena();
        let meta = crate::publish::StreamMeta {
            stream: StreamId(7),
            format: PayloadFormat::EncodedVideo,
            shape_name: "H264AnnexB".into(),
            anchor: crate::clock::Anchor {
                boot: BootId([1; 16]),
                mono_ns: 1_000_000_000,
                utc_ns: 1_800_000_000_000_000_000,
                bracket_ns: 7,
            },
            residual_offset_ns: 0,
        };
        let mut publisher = crate::video::EncodedPublisher {
            stream: crate::publish::VideoStream::new(meta, arena.0.clone(), 0, 42, None),
            pipeline,
            slot_bytes: 256 * 1024,
        };
        let picture = jpeg(W, H);
        let mut notices = Notices::default();
        let mut decoder = continuum_tier::video::H264StreamDecoder::new().unwrap();
        for (index, mono) in [1_900_000_000, 1_950_000_000, 2_000_000_000, 2_050_000_000].into_iter().enumerate() {
            if index == 3 {
                publisher.pipeline.break_chain();
            }
            let sample = Capture { width: u32::from(W), height: u32::from(H), ..capture(mono, 10 + index as u64, 23) };
            let published = publisher.publish_frame(sample, &picture, &mut notices).unwrap();
            assert_eq!(publisher.pipeline.backend_name(), "h264_nvenc", "runtime fallback cannot validate NVENC");
            let bytes = published_bytes(&arena.0, published);
            assert!(!bytes.starts_with(&[0xff, 0xd8]));
            if index != 1 {
                assert!(
                    continuum_tier::gop::h264_idr(&bytes),
                    "initial, two-second cut, and explicit break require IDR"
                );
                let nal_types: Vec<_> = bytes.windows(4).filter(|w| w[..3] == [0, 0, 1]).map(|w| w[3] & 31).collect();
                assert!(nal_types.contains(&7) && nal_types.contains(&8), "independent cuts require SPS and PPS");
                let mut independent = continuum_tier::video::H264StreamDecoder::new().unwrap();
                assert_eq!(independent.push(&bytes, sample.device_seq).unwrap().len(), 1);
            }
            let decoded = decoder.push(&bytes, sample.device_seq).unwrap();
            assert_eq!(decoded.len(), 1, "each JPEG must produce one immediate decodable AU");
            assert_eq!(
                (decoded[0].tag, decoded[0].width, decoded[0].height),
                (sample.device_seq, u32::from(W), u32::from(H))
            );
            let header = &notices.0[index].0;
            let native = header.header();
            let cap = native.capture().unwrap();
            let start = 1_800_000_000_000_000_000 + mono - 1_000_000_000;
            assert_eq!((header.start_ns(), native.end_ns()), (start, start + sample.duration_ns));
            assert_eq!(native.seq(), sample.device_seq as u32);
            assert_eq!(
                (cap.capture_mono_ns(), cap.device_seq(), cap.device_ts(), cap.arrival_ns()),
                (sample.mono_ns, sample.device_seq, sample.device_ts, sample.arrival_ns)
            );
            assert_eq!(
                (cap.width(), cap.height(), cap.flip(), cap.capture_epoch()),
                (u32::from(W), u32::from(H), sample.flip, sample.capture_epoch)
            );
            arena.0.mark_flushed(published.slot).unwrap();
            arena.0.reclaim(published.slot).unwrap();
        }
        assert_eq!(notices.0.len(), 4);
    }
}
