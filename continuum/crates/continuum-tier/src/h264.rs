//! The H.264 encoder: a raw camera's Tier 1 video and capture's camera encoding.
//!
//! [`H264Encoder`] takes decoded frames (NV12, full-range BT.601, as [`crate::hydrate::rgb_to_nv12`] makes) with
//! their times and returns H.264 access units in Annex B form, ready for WebCodecs' `annexb` decoder config. What the
//! gateway relies on (the encoder core is in `encode.rs`):
//!
//! - Every keyframe carries its SPS and PPS in-band (no global header), so the gateway can start a browser on any
//!   keyframe and read the codec string from it.
//! - No B-frames: access units come out in presentation order, one per frame, each with its frame's own times.
//!   This is checked, not assumed: every packet's pts and dts must be the frame it answers, so an encoder that
//!   reorders anyway is an error rather than a stream with shuffled times.
//! - The stream says what its pixels are: full range, BT.601 matrix, primaries, and transfer (SMPTE 170M), as
//!   `rgb_to_nv12` makes them, so a browser doesn't guess BT.709 for larger frames.
//!
//! An encoder failure partway through a range (a lost GPU, say) is returned as an error from `encode` or `finish`,
//! never an early, silent end of the stream.
//!
//! The encoder is the first of [`H264_ENCODERS`] that opens on this machine: NVENC on Linux with an NVIDIA GPU,
//! VideoToolbox on the Mac, and a software encoder everywhere else ([`SOFTWARE_H264_ENCODERS`]). Main profile (High
//! from OpenH264, which has no Main), which every browser decodes.

use crate::TierError;
pub use crate::encode::AccessUnit;
use crate::encode::{Codec, Encoder, Settings};

/// The encoders tried, in order of preference.
pub const H264_ENCODERS: &[&str] = &["h264_nvenc", "h264_videotoolbox", "libx264", "libopenh264"];

/// The software encoders, in order of preference: libx264 where FFmpeg has it (a distribution's GPL build), and
/// OpenH264 in the wheels' private LGPL build, which can't carry libx264 (docs/CONTINUUM.md, "Wheels").
pub const SOFTWARE_H264_ENCODERS: &[&str] = &["libx264", "libopenh264"];

/// How the browser stream is encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct H264Settings {
    /// Frames from one keyframe to the next: where a browser can start, and how far a seek has to decode.
    pub gop_frames: u32,
    /// The nominal frame rate, for the encoder's rate control (the access units keep each frame's real times).
    pub frame_rate: u32,
    /// Target bits per second.
    pub bit_rate: usize,
    /// Full range (pictures made from JPEG) rather than limited (a camera's raw video, kept as it came).
    pub full_range: bool,
    /// Every frame's access unit as soon as it's encoded: no lookahead or frame threads holding frames back (Tier 1
    /// of a raw camera, which has to be durable within seconds).
    pub low_delay: bool,
    /// Threads for a software encoder: 0 lets it choose. Playback transcoding uses 1.
    pub threads: u32,
}

impl Default for H264Settings {
    fn default() -> Self {
        Self { gop_frames: 30, frame_rate: 15, bit_rate: 2_000_000, full_range: true, low_delay: false, threads: 0 }
    }
}

/// An H.264 encoder for one range of one stream.
pub struct H264Encoder(Encoder);

impl H264Encoder {
    /// Opens the first encoder in [`H264_ENCODERS`] that works here.
    ///
    /// # Errors
    ///
    /// [`TierError::Codec`] when none opens, or for odd or zero dimensions.
    pub fn open(width: u32, height: u32, settings: H264Settings) -> Result<Self, TierError> {
        Self::open_with(H264_ENCODERS, width, height, settings)
    }

    /// Opens the first of `names` that works here.
    ///
    /// # Errors
    ///
    /// [`TierError::Codec`] when none opens (the message names each one's error), or for odd or zero dimensions.
    pub fn open_with(names: &[&str], width: u32, height: u32, settings: H264Settings) -> Result<Self, TierError> {
        let s = Settings {
            gop_frames: settings.gop_frames,
            frame_rate: settings.frame_rate,
            bit_rate: settings.bit_rate,
            full_range: settings.full_range,
            low_delay: settings.low_delay,
            threads: settings.threads,
        };
        Encoder::open_with(Codec::H264, names, width, height, s).map(Self)
    }

    /// Opens only NVENC on the selected GPU at constant quality. Failure never selects a CPU encoder.
    /// # Errors
    /// Invalid settings, unsupported NVENC, or an unavailable GPU.
    pub fn open_nvenc(width: u32, height: u32, settings: H264Settings, gpu: u32, cq: u32) -> Result<Self, TierError> {
        Encoder::open_nvenc(
            width,
            height,
            Settings {
                gop_frames: settings.gop_frames,
                frame_rate: settings.frame_rate,
                bit_rate: 0,
                full_range: settings.full_range,
                low_delay: true,
                threads: 1,
            },
            gpu,
            cq,
        )
        .map(Self)
    }

    /// The encoder in use, like `h264_nvenc`.
    #[must_use]
    pub fn encoder_name(&self) -> &str {
        self.0.name()
    }

    /// Encodes one NV12 frame and returns the access units ready so far (an encoder may hold a few back).
    ///
    /// # Errors
    ///
    /// [`TierError::Codec`] for a buffer of the wrong size, backwards times, or an encoder failure.
    pub fn encode(&mut self, nv12: &[u8], start_ns: u64, end_ns: u64) -> Result<Vec<AccessUnit>, TierError> {
        self.0.encode(nv12, start_ns, end_ns)
    }

    /// Encodes a frame with an explicit independently decodable keyframe request at a capture-time cut.
    /// # Errors
    /// The same input and encoder errors as [`Self::encode`].
    pub fn encode_keyframe(
        &mut self,
        nv12: &[u8],
        start_ns: u64,
        end_ns: u64,
        key: bool,
    ) -> Result<Vec<AccessUnit>, TierError> {
        self.0.encode_keyframe(nv12, start_ns, end_ns, key)
    }

    /// Flushes the encoder and returns the access units it held back.
    ///
    /// # Errors
    ///
    /// [`TierError::Codec`] on an encoder failure, or if a frame it was given never came back.
    pub fn finish(self) -> Result<Vec<AccessUnit>, TierError> {
        self.0.finish()
    }
}
