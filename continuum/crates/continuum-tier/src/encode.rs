//! The H.264 encoder core behind [`crate::h264`] (the `ffmpeg` feature), and the AV1 one behind the archive pass
//! ([`crate::av1`]).
//!
//! NV12 frames (full-range BT.601, as [`crate::hydrate::rgb_to_nv12`] makes them) with their times go in; Annex B
//! access units come out, one per frame and in order. What it promises:
//!
//! - Parameter sets in-band on every keyframe (no global header), so any keyframe starts a decoder on its own:
//!   its SPS and PPS.
//! - No B-frames, and this is checked: every packet's pts and dts must be the frame it answers, so an encoder that
//!   reorders anyway is an error, never a stream with shuffled times.
//! - Keyframes exactly every `gop_frames` (scene cuts off), so a seek always decodes the same span.
//! - The stream signals BT.601 (SMPTE 170M) matrix, primaries, and transfer, and full range (pictures made from JPEG)
//!   or limited range (a camera's raw video, kept as it came) as the settings say.
//! - An encoder failure partway through is an error from `encode` or `finish`, never an early end.
//!
//! The raw-camera loop and the readers that decode video (frame grabs, export, and the gateway) can open codecs
//! concurrently. Their native initialization and open operations share [`codec_open`]; frame processing remains
//! independent after the guard is dropped. FFmpeg initialization runs once because ffmpeg-next rewrites a global
//! error-string table on each initialization, including while another thread could be formatting an error.
//!
//! Closing the first of several overlapping NVENC sessions can crash the process. The native owner
//! retains that first session if its caller finishes early, then closes it after the remaining sessions close.
//! This holds at most one retired native context (and its NVENC session allocation), including across replacement
//! camera opens. Software and VideoToolbox sessions keep their ordinary lifetime; frame processing stays parallel.

use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};
use std::sync::{Mutex, MutexGuard, OnceLock};

use ff::codec::{self, encoder};
use ff::format::Pixel;
use ff::util::color::{Primaries, Range, Space, TransferCharacteristic};
use ff::{Dictionary, Packet, Rational, frame};
use ffmpeg_next as ff;

use crate::TierError;

/// Initializes FFmpeg once and serializes native opening and encoder closing within this process. Callers retain
/// the guard through native lifecycle operations only, never frame processing, and must not acquire it recursively.
///
/// FFmpeg's own log is set to errors only first: it writes to stderr with no stamp or level, and its warnings (the
/// H.264 parser's "Overread VUI by 8 bits" for every camera stream opened, swscaler's "deprecated pixel format")
/// were most of see-decode's log (#59). A failure still comes back to the caller as an error.
pub(crate) fn codec_open() -> Result<MutexGuard<'static, CodecLifecycle>, ff::Error> {
    static OPEN: Mutex<CodecLifecycle> = Mutex::new(CodecLifecycle { nvenc_live: 0, retired_first: None });
    static INITIALIZED: OnceLock<Result<(), ff::Error>> = OnceLock::new();

    let guard = OPEN.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    (*INITIALIZED.get_or_init(|| {
        ff::util::log::set_level(ff::util::log::Level::Error);
        ff::init()
    }))?;
    Ok(guard)
}

/// Native opens and closes share this lock. NVENC's first session stays alive while later sessions overlap it.
pub(crate) struct CodecLifecycle {
    nvenc_live: usize,
    retired_first: Option<encoder::Video>,
}

/// Owns the native context, including early returns and an encoder dropped without `finish`.
struct NativeEncoder {
    video: Option<encoder::Video>,
    nvenc: bool,
    first: bool,
}

impl NativeEncoder {
    /// Called with the open lock already held, only after native initialization succeeds.
    fn new(video: encoder::Video, name: &str, lifecycle: &mut CodecLifecycle) -> Self {
        let nvenc = name == "h264_nvenc";
        let first = nvenc && lifecycle.nvenc_live == 0;
        if nvenc {
            lifecycle.nvenc_live += 1;
        }
        Self { video: Some(video), nvenc, first }
    }
}

impl Deref for NativeEncoder {
    type Target = encoder::Video;

    fn deref(&self) -> &Self::Target {
        self.video.as_ref().expect("native context is present until drop")
    }
}

impl DerefMut for NativeEncoder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.video.as_mut().expect("native context is present until drop")
    }
}

impl Drop for NativeEncoder {
    fn drop(&mut self) {
        // Initialization already succeeded before this owner was created. No frame-processing lock is needed.
        let mut lifecycle = codec_open().expect("an opened encoder initialized FFmpeg");
        let video = self.video.take();
        if self.nvenc {
            lifecycle.nvenc_live -= 1;
            if self.first && lifecycle.nvenc_live != 0 {
                lifecycle.retired_first = video;
            } else {
                drop(video);
            }
            if lifecycle.nvenc_live == 0 {
                drop(lifecycle.retired_first.take());
            }
        } else {
            drop(video);
        }
    }
}

/// One encoded frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccessUnit {
    /// The access unit in Annex B form (start codes); a keyframe's begins with its parameter sets.
    pub bytes: Vec<u8>,
    /// The frame's start time, UTC nanoseconds.
    pub start_ns: u64,
    /// Its end time.
    pub end_ns: u64,
    /// A keyframe: decoding can start here.
    pub key: bool,
}

/// The codec: H.264 for a raw camera's Tier 1, AV1 for the opt-in archive pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Codec {
    H264,
    Av1,
}

impl Codec {
    fn label(self) -> &'static str {
        match self {
            Self::H264 => "h264",
            Self::Av1 => "av1",
        }
    }

    /// The encoder's private options for one backend: keyframes exactly every `gop` frames, no B-frames, and
    /// parameter sets in-band; with `low_delay`, every frame's packet comes out as soon as it's encoded (no lookahead
    /// or frame threads holding frames back).
    fn options(self, name: &str, gop: u32, low_delay: bool, threads: u32) -> Dictionary<'static> {
        let mut o = Dictionary::new();
        if self == Self::Av1 {
            return o; // libsvtav1's controls come from Av1Options
        }
        o.set("profile", "main");
        match name {
            "libx264" => {
                o.set("preset", "veryfast");
                o.set("threads", &if threads == 0 { "auto".to_owned() } else { threads.to_string() });
                if low_delay {
                    o.set("tune", "zerolatency"); // no lookahead or frame threads: each frame's packet comes out with it
                }
                o.set("x264-params", &format!("bframes=0:scenecut=0:keyint={gop}:min-keyint={gop}"));
            }
            "h264_nvenc" => {
                o.set("preset", "p4");
                o.set("bf", "0");
                o.set("no-scenecut", "1");
                if low_delay {
                    o.set("zerolatency", "1");
                    o.set("delay", "0");
                    o.set("rc-lookahead", "0");
                }
            }
            "h264_videotoolbox" if low_delay => {
                o.set("realtime", "1");
            }
            _ => {}
        }
        o
    }
}

/// The color signaling an encoder writes: a raw camera's is BT.601 ([`Colors::bt601`]); the archive pass copies its
/// source's, so re-encoding never reinterprets the picture's colors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Colors {
    pub range: Range,
    pub space: Space,
    pub primaries: Primaries,
    pub transfer: TransferCharacteristic,
}

impl Colors {
    pub(crate) fn bt601(full_range: bool) -> Self {
        Self {
            range: if full_range { Range::JPEG } else { Range::MPEG },
            space: Space::SMPTE170M,
            primaries: Primaries::SMPTE170M,
            transfer: TransferCharacteristic::SMPTE170M,
        }
    }
}

/// The SVT-AV1 controls the archive pass sets (docs/STORAGE.md: preset 8, CRF 52 by default).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Av1Options {
    pub preset: u8,
    pub crf: u8,
    /// SVT-AV1's level of parallelism (`lp`): about how many cores it keeps busy.
    pub parallelism: u32,
}

/// How a stream is encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Settings {
    pub gop_frames: u32,
    pub frame_rate: u32,
    pub bit_rate: usize,
    /// Full range (pictures from JPEG) rather than limited (a camera's raw video).
    pub full_range: bool,
    /// Every frame's packet as soon as it's encoded.
    pub low_delay: bool,
    /// Encoder threads for software encoders: 0 lets the encoder choose.
    pub threads: u32,
}

/// How [`Encoder::try_open`] configures the encoder beyond its settings.
#[derive(Clone, Copy, Debug)]
enum Open {
    Plain,
    Nvenc { gpu: u32, cq: u32 },
    Av1 { options: Av1Options, colors: Colors },
}

/// A video encoder for one range of one stream.
pub(crate) struct Encoder {
    codec: Codec,
    encoder: NativeEncoder,
    name: String,
    width: u32,
    height: u32,
    /// What the encoder takes: NV12 where it can, else planar 4:2:0 (libx265), filled from the NV12 given.
    pixel: Pixel,
    /// The range the frames are in, signalled on each.
    range: Range,
    next_pts: i64,
    /// Frames sent and not yet returned, in order: their pts and times (no B-frames, so packets come back in this
    /// order, which `drain` checks).
    pending: VecDeque<(i64, u64, u64)>,
}

fn codec_err(codec: Codec, what: &str, e: ff::Error) -> TierError {
    TierError::Codec(format!("{} {what}: {e}", codec.label()))
}

impl Encoder {
    /// Opens the first of `names` that works here.
    pub(crate) fn open_with(
        codec: Codec,
        names: &[&str],
        width: u32,
        height: u32,
        settings: Settings,
    ) -> Result<Self, TierError> {
        let label = codec.label();
        if width == 0 || height == 0 || width % 2 == 1 || height % 2 == 1 {
            return Err(TierError::Codec(format!("{label} needs even, non-zero dimensions, not {width}x{height}")));
        }
        if settings.gop_frames == 0 || settings.frame_rate == 0 {
            return Err(TierError::Codec(format!("{label} needs a non-zero GOP and frame rate")));
        }
        let mut lifecycle = codec_open().map_err(|e| codec_err(codec, "init", e))?;
        let mut tried = Vec::new();
        for &name in names {
            match Self::try_open(codec, name, width, height, settings, Open::Plain, &mut lifecycle) {
                Ok(encoder) => return Ok(encoder),
                Err(e) => tried.push(format!("{name}: {e}")),
            }
        }
        Err(TierError::Codec(format!("no {label} encoder opened ({})", tried.join("; "))))
    }

    pub(crate) fn open_nvenc(
        width: u32,
        height: u32,
        settings: Settings,
        gpu: u32,
        cq: u32,
    ) -> Result<Self, TierError> {
        if gpu > i32::MAX as u32
            || cq > 51
            || width == 0
            || height == 0
            || !width.is_multiple_of(2)
            || !height.is_multiple_of(2)
            || settings.gop_frames == 0
            || settings.frame_rate == 0
        {
            return Err(TierError::Codec(
                "NVENC requires valid dimensions, frame rate, GPU ordinal and CQ 0..51".into(),
            ));
        }
        let mut lifecycle = codec_open().map_err(|e| codec_err(Codec::H264, "init", e))?;
        Self::try_open(Codec::H264, "h264_nvenc", width, height, settings, Open::Nvenc { gpu, cq }, &mut lifecycle)
    }

    /// An SVT-AV1 encoder for one archived chunk: `settings.gop_frames` frames per keyframe, constant quality, and
    /// the source's `colors`.
    pub(crate) fn open_av1(
        width: u32,
        height: u32,
        settings: Settings,
        options: Av1Options,
        colors: Colors,
    ) -> Result<Self, TierError> {
        if width == 0 || height == 0 || width % 2 == 1 || height % 2 == 1 {
            return Err(TierError::Codec(format!("av1 needs even, non-zero dimensions, not {width}x{height}")));
        }
        if settings.gop_frames == 0 || settings.frame_rate == 0 || options.preset > 13 || options.crf > 63 {
            return Err(TierError::Codec("av1 needs a GOP, a frame rate, preset 0..13, and CRF 0..63".into()));
        }
        let mut lifecycle = codec_open().map_err(|e| codec_err(Codec::Av1, "init", e))?;
        Self::try_open(Codec::Av1, "libsvtav1", width, height, settings, Open::Av1 { options, colors }, &mut lifecycle)
    }

    fn try_open(
        codec: Codec,
        name: &str,
        width: u32,
        height: u32,
        settings: Settings,
        open: Open,
        lifecycle: &mut CodecLifecycle,
    ) -> Result<Self, TierError> {
        let found = encoder::find_by_name(name).ok_or_else(|| TierError::Codec("not in this FFmpeg".to_owned()))?;
        let mut video = codec::context::Context::new_with_codec(found)
            .encoder()
            .video()
            .map_err(|e| codec_err(codec, "context", e))?;
        let takes_nv12 = found.video().ok().and_then(|v| v.formats()).is_some_and(|mut f| f.any(|p| p == Pixel::NV12));
        let pixel = if takes_nv12 { Pixel::NV12 } else { Pixel::YUV420P };
        video.set_width(width);
        video.set_height(height);
        video.set_format(pixel);
        video.set_time_base(Rational(1, settings.frame_rate as i32));
        video.set_frame_rate(Some(Rational(settings.frame_rate as i32, 1)));
        video.set_gop(settings.gop_frames);
        video.set_max_b_frames(0);
        video.set_bit_rate(settings.bit_rate);
        let colors = match open {
            Open::Av1 { colors, .. } => colors,
            _ => Colors::bt601(settings.full_range),
        };
        let range = colors.range;
        video.set_color_range(range);
        video.set_colorspace(colors.space);
        video.set_color_primaries(colors.primaries);
        video.set_color_transfer_characteristic(colors.transfer);
        let mut options = codec.options(name, settings.gop_frames, settings.low_delay, settings.threads);
        match open {
            Open::Plain => {}
            Open::Nvenc { gpu, cq } => {
                options.set("gpu", &gpu.to_string());
                options.set("preset", "p5");
                options.set("rc", "vbr");
                options.set("cq", &cq.to_string());
                options.set("forced-idr", "1");
            }
            Open::Av1 { options: av1, .. } => {
                options.set("preset", &av1.preset.to_string());
                options.set("crf", &av1.crf.to_string());
                options.set("svtav1-params", &format!("lp={}", av1.parallelism.max(1)));
            }
        }
        let encoder = video.open_as_with(found, options).map_err(|e| codec_err(codec, "open", e))?;
        Ok(Self {
            codec,
            encoder: NativeEncoder::new(encoder, name, lifecycle),
            name: name.to_owned(),
            width,
            height,
            pixel,
            range,
            next_pts: 0,
            pending: VecDeque::new(),
        })
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn encode(&mut self, nv12: &[u8], start_ns: u64, end_ns: u64) -> Result<Vec<AccessUnit>, TierError> {
        self.encode_keyframe(nv12, start_ns, end_ns, false)
    }

    pub(crate) fn encode_keyframe(
        &mut self,
        nv12: &[u8],
        start_ns: u64,
        end_ns: u64,
        key: bool,
    ) -> Result<Vec<AccessUnit>, TierError> {
        let (w, h) = (self.width as usize, self.height as usize);
        if nv12.len() != w * h * 3 / 2 {
            return Err(TierError::Codec(format!(
                "an NV12 {w}x{h} frame is {} bytes, not {}",
                w * h * 3 / 2,
                nv12.len()
            )));
        }
        if end_ns < start_ns {
            return Err(TierError::Codec("a frame ends before it starts".to_owned()));
        }
        let mut frame = frame::Video::new(self.pixel, self.width, self.height);
        let stride = frame.stride(0);
        let luma = frame.data_mut(0);
        for row in 0..h {
            luma[row * stride..row * stride + w].copy_from_slice(&nv12[row * w..(row + 1) * w]);
        }
        let chroma = &nv12[w * h..];
        if self.pixel == Pixel::NV12 {
            let stride = frame.stride(1);
            let uv = frame.data_mut(1);
            for row in 0..h / 2 {
                uv[row * stride..row * stride + w].copy_from_slice(&chroma[row * w..(row + 1) * w]);
            }
        } else {
            // Planar: NV12's interleaved U and V go to their own planes.
            for (plane, pick) in [(1, 0), (2, 1)] {
                let stride = frame.stride(plane);
                let dst = frame.data_mut(plane);
                for row in 0..h / 2 {
                    for col in 0..w / 2 {
                        dst[row * stride + col] = chroma[row * w + 2 * col + pick];
                    }
                }
            }
        }
        frame.set_color_range(self.range);
        frame.set_pts(Some(self.next_pts));
        if key {
            frame.set_kind(ff::picture::Type::I);
        }
        // Record the frame only once the encoder has taken it, so a refused send can't shift later frames' times.
        self.encoder.send_frame(&frame).map_err(|e| codec_err(self.codec, "send", e))?;
        self.pending.push_back((self.next_pts, start_ns, end_ns));
        self.next_pts += 1;
        self.drain()
    }

    pub(crate) fn finish(mut self) -> Result<Vec<AccessUnit>, TierError> {
        self.encoder.send_eof().map_err(|e| codec_err(self.codec, "eof", e))?;
        let out = self.drain()?;
        if !self.pending.is_empty() {
            return Err(TierError::Codec(format!(
                "{} encoder dropped {} frames",
                self.codec.label(),
                self.pending.len()
            )));
        }
        Ok(out)
    }

    /// The packets the encoder has ready. Stops when it wants more input or has finished; any other error is
    /// returned, so a failure never looks like the end of the range.
    fn drain(&mut self) -> Result<Vec<AccessUnit>, TierError> {
        let label = self.codec.label();
        let mut out = Vec::new();
        let mut packet = Packet::empty();
        loop {
            match self.encoder.receive_packet(&mut packet) {
                Ok(()) => {}
                Err(ff::Error::Other { errno: ff::util::error::EAGAIN } | ff::Error::Eof) => return Ok(out),
                Err(e) => return Err(codec_err(self.codec, "encode", e)),
            }
            let (pts, start_ns, end_ns) = self
                .pending
                .pop_front()
                .ok_or_else(|| TierError::Codec(format!("{label} encoder returned more frames than it was given")))?;
            if packet.pts() != Some(pts) || packet.dts() != Some(pts) {
                return Err(TierError::Codec(format!(
                    "{label} encoder reordered frames (expected pts and dts {pts}, got {:?} and {:?})",
                    packet.pts(),
                    packet.dts()
                )));
            }
            let bytes = packet.data().unwrap_or_default().to_vec();
            out.push(AccessUnit { bytes, start_ns, end_ns, key: packet.is_key() });
        }
    }
}
