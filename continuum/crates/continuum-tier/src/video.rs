//! Video decoding, behind the `ffmpeg` feature.
//!
//! [`decode_gop`] turns one GOP's units ([`crate::gop`]) back into NV12 frames with their times: a raw camera's
//! H.264 access units ([`crate::ChunkFormat::H264Au`]). [`H264StreamDecoder`] decodes a live H.264 stream, on the GPU
//! where it can.

use ff::format::Pixel;
use ffmpeg_next as ff;

use crate::TierError;
use crate::gop::Unit;

/// One decoded frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedFrame {
    /// Its start time, UTC nanoseconds.
    pub start_ns: u64,
    /// Its end time.
    pub end_ns: u64,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The picture as NV12 (the format the doc names for decoded frames).
    pub nv12: Vec<u8>,
}

fn decode_err(what: &str, e: ff::Error) -> TierError {
    TierError::Codec(format!("decode {what}: {e}"))
}

/// YUVJ420P has the same planes as YUV420P, with full range encoded in its deprecated format tag. Retag the owned
/// decoded frame before consumers create a scaler; its pixel buffers and signaled color matrix stay unchanged.
pub(crate) fn normalize_decoded_range(picture: &mut ff::frame::Video) {
    if picture.format() == Pixel::YUVJ420P {
        picture.set_format(Pixel::YUV420P);
        picture.set_color_range(ff::util::color::Range::JPEG);
    }
}

/// The NV12 bytes of a decoded picture (4:2:0 planar or already NV12), with no padding.
pub(crate) fn to_nv12(f: &ff::frame::Video) -> Result<Vec<u8>, TierError> {
    let (w, h) = (f.width() as usize, f.height() as usize);
    let mut out = Vec::with_capacity(w * h * 3 / 2);
    let rows = |plane: usize, width: usize, height: usize, out: &mut Vec<u8>| {
        let (stride, data) = (f.stride(plane), f.data(plane));
        for row in 0..height {
            out.extend_from_slice(&data[row * stride..row * stride + width]);
        }
    };
    match f.format() {
        Pixel::NV12 => {
            rows(0, w, h, &mut out);
            rows(1, w, h / 2, &mut out);
        }
        Pixel::YUV420P | Pixel::YUVJ420P => {
            rows(0, w, h, &mut out);
            let (us, vs, u, v) = (f.stride(1), f.stride(2), f.data(1), f.data(2));
            for row in 0..h / 2 {
                for col in 0..w / 2 {
                    out.push(u[row * us + col]);
                    out.push(v[row * vs + col]);
                }
            }
        }
        other => return Err(TierError::Codec(format!("decode: unexpected pixel format {other:?}"))),
    }
    Ok(out)
}

/// The decoder for a video chunk's units: H.264 for [`crate::ChunkFormat::H264Au`], AV1 for the archive pass's
/// [`crate::ChunkFormat::Av1Au`] (#1479), `None` for any other format.
#[must_use]
pub fn codec_of(format: crate::chunk::ChunkFormat) -> Option<ff::codec::Id> {
    match format {
        crate::chunk::ChunkFormat::H264Au => Some(ff::codec::Id::H264),
        crate::chunk::ChunkFormat::Av1Au => Some(ff::codec::Id::AV1),
        _ => None,
    }
}

/// Decodes one H.264 GOP's units into NV12 frames, in presentation order, each with its unit's times.
///
/// # Errors
///
/// [`TierError::Codec`] for a GOP that doesn't start with a keyframe, a decoder failure, or a frame the decoder
/// returned that wasn't sent.
pub fn decode_gop(units: &[Unit]) -> Result<Vec<DecodedFrame>, TierError> {
    decode_units(ff::codec::Id::H264, units)
}

/// As [`decode_gop`], for a GOP of any FFmpeg `codec` (v1's recordings are H.264): `units` in decode order, each with
/// its own presentation time, and the frames come back in presentation order with those times.
///
/// # Errors
///
/// As [`decode_gop`].
pub fn decode_units(codec: ff::codec::Id, units: &[Unit]) -> Result<Vec<DecodedFrame>, TierError> {
    decode_units_with(codec, units, to_nv12)
}

fn decode_units_with(
    codec: ff::codec::Id,
    units: &[Unit],
    mut pixels: impl FnMut(&ff::frame::Video) -> Result<Vec<u8>, TierError>,
) -> Result<Vec<DecodedFrame>, TierError> {
    if !units.first().is_some_and(|u| u.key) {
        return Err(TierError::Codec("decode: a GOP must start with its keyframe".to_owned()));
    }
    let mut decoder = {
        let _open = crate::encode::codec_open().map_err(|e| decode_err("init", e))?;
        let found = ff::decoder::find(codec).ok_or_else(|| TierError::Codec(format!("no {codec:?} decoder")))?;
        ff::codec::context::Context::new_with_codec(found).decoder().video().map_err(|e| decode_err("open", e))?
    };
    let mut out = Vec::with_capacity(units.len());
    let mut picture = ff::frame::Video::empty();
    let mut take = |decoder: &mut ff::decoder::Video, out: &mut Vec<DecodedFrame>| -> Result<(), TierError> {
        loop {
            match decoder.receive_frame(&mut picture) {
                Ok(()) => {}
                Err(ff::Error::Other { errno: ff::util::error::EAGAIN } | ff::Error::Eof) => return Ok(()),
                Err(e) => return Err(decode_err("frame", e)),
            }
            normalize_decoded_range(&mut picture);
            let index = picture.pts().and_then(|p| usize::try_from(p).ok());
            let unit = index
                .and_then(|i| units.get(i))
                .ok_or_else(|| TierError::Codec("decode: a frame that wasn't sent".to_owned()))?;
            out.push(DecodedFrame {
                start_ns: unit.start_ns,
                end_ns: unit.end_ns,
                width: picture.width(),
                height: picture.height(),
                nv12: pixels(&picture)?,
            });
        }
    };
    for (i, unit) in units.iter().enumerate() {
        let mut packet = ff::Packet::copy(&unit.bytes);
        packet.set_pts(Some(i as i64));
        decoder.send_packet(&packet).map_err(|e| decode_err("send", e))?;
        take(&mut decoder, &mut out)?;
    }
    decoder.send_eof().map_err(|e| decode_err("eof", e))?;
    take(&mut decoder, &mut out)?;
    out.sort_by_key(|f| f.start_ns);
    Ok(out)
}

/// Converts using explicit decoded YUV colorimetry. swscale's default matrix
/// cannot be used here: it would silently turn native BT.709 into BT.601.
fn signaled_rgb(picture: &ff::frame::Video) -> Result<crate::hydrate::Picture, TierError> {
    use ff::software::scaling::{Context, Flags};
    use ff::util::color::{Range, Space};
    let matrix = match picture.color_space() {
        Space::BT709 => ff::ffi::SWS_CS_ITU709,
        Space::FCC => ff::ffi::SWS_CS_FCC,
        Space::BT470BG | Space::SMPTE170M => ff::ffi::SWS_CS_ITU601,
        Space::SMPTE240M => ff::ffi::SWS_CS_SMPTE240M,
        Space::BT2020NCL => ff::ffi::SWS_CS_BT2020,
        other => return Err(TierError::Codec(format!("RGB decode: unsupported color matrix {other:?}"))),
    };
    let full = match picture.color_range() {
        Range::JPEG => 1,
        Range::MPEG => 0,
        other => return Err(TierError::Codec(format!("RGB decode: unsupported color range {other:?}"))),
    };
    let (width, height) = (picture.width(), picture.height());
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > 16_000_000 {
        return Err(TierError::Codec("RGB decode: pixel limit exceeded".into()));
    }
    let mut scaler = Context::get(picture.format(), width, height, Pixel::RGB24, width, height, Flags::BILINEAR)
        .map_err(|e| decode_err("RGB scaler", e))?;
    configure_rgb_colors(&mut scaler, matrix as i32, full)?;
    let mut rgb = ff::frame::Video::empty();
    scaler.run(picture, &mut rgb).map_err(|e| decode_err("RGB scale", e))?;
    let row_bytes = width as usize * 3;
    let mut packed = Vec::with_capacity(row_bytes * height as usize);
    for row in 0..height as usize {
        let offset = row * rgb.stride(0);
        packed.extend_from_slice(&rgb.data(0)[offset..offset + row_bytes]);
    }
    Ok(crate::hydrate::Picture { width, height, rgb: packed })
}

/// ffmpeg-next has no safe colorspace-details setter. Keep the FFI lifetime
/// boundary here, with one owned scaler and FFmpeg's static coefficient table.
fn configure_rgb_colors(scaler: &mut ff::software::scaling::Context, matrix: i32, full: i32) -> Result<(), TierError> {
    // SAFETY: matrix is a supported SWS_CS constant. The returned coefficients
    // are a static FFmpeg table; scaler owns a live, non-null SwsContext for the
    // entire call, and exclusive access prevents concurrent mutation.
    let result = unsafe {
        let coefficients = ff::ffi::sws_getCoefficients(matrix);
        if coefficients.is_null() {
            return Err(TierError::Codec("RGB decode: missing color coefficients".into()));
        }
        ff::ffi::sws_setColorspaceDetails(scaler.as_mut_ptr(), coefficients, full, coefficients, 1, 0, 1 << 16, 1 << 16)
    };
    if result < 0 {
        return Err(TierError::Codec(format!("RGB decode: cannot configure color conversion ({result})")));
    }
    Ok(())
}

/// One picture from a stateful H.264 decoder. `tag` is the caller's opaque input tag, never a guessed timestamp.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaggedFrame {
    /// Tag from the access unit that produced this picture, even when B-frames reorder the output.
    pub tag: u64,
    /// Decoded dimensions.
    pub width: u32,
    /// Decoded height.
    pub height: u32,
    /// Packed NV12 pixels with no row padding.
    pub nv12: Vec<u8>,
    /// Actual decoded YUV range: full, limited, unspecified, or unsupported.
    pub color_range: &'static str,
    /// Actual decoded matrix; BT.470BG/SMPTE170M are represented as bt601.
    pub color_matrix: &'static str,
    /// Optional packed BGR converted using explicit supported color signaling, never guessed.
    pub bgr: Option<Vec<u8>>,
}

fn live_color_outputs(
    picture: &ff::frame::Video,
    want_bgr: bool,
) -> Result<(&'static str, &'static str, Option<Vec<u8>>), TierError> {
    use ff::util::color::{Range, Space};
    let range = match picture.color_range() {
        Range::JPEG => "full",
        Range::MPEG => "limited",
        Range::Unspecified => "unspecified",
    };
    let matrix = match picture.color_space() {
        Space::BT470BG | Space::SMPTE170M => "bt601",
        Space::BT709 => "bt709",
        Space::BT2020NCL => "bt2020nc",
        other => other.name().unwrap_or("unspecified"),
    };
    let supported = matches!(picture.color_range(), Range::JPEG | Range::MPEG)
        && matches!(
            picture.color_space(),
            Space::BT709 | Space::FCC | Space::BT470BG | Space::SMPTE170M | Space::SMPTE240M | Space::BT2020NCL
        );
    let bgr = if want_bgr && supported {
        let mut pixels = signaled_rgb(picture)?.rgb;
        for pixel in pixels.chunks_exact_mut(3) {
            pixel.swap(0, 2);
        }
        Some(pixels)
    } else {
        None
    };
    Ok((range, matrix, bgr))
}

/// Backend selected explicitly for a live H.264 decoder. CUDA output is downloaded to host NV12;
/// this is not CUDA IPC or a GPU-resident frame interface.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum H264DecodeBackend {
    /// Portable single-threaded FFmpeg software decoding.
    #[default]
    Software,
    /// FFmpeg's native H.264 parser with NVDEC on this CUDA ordinal. Never falls back to software.
    Cuda {
        /// CUDA-visible device ordinal, not a physical PCI identifier.
        gpu: u32,
    },
}

#[derive(Default)]
struct CudaTags {
    pending: std::collections::BTreeMap<i64, u64>,
    displayed: Option<i64>,
    dropped: Vec<u64>,
}

fn cuda_tick(ns: i64) -> i64 {
    let wide = i128::from(ns);
    let magnitude = (wide.abs() + 50) / 100;
    (if wide < 0 { -magnitude } else { magnitude }) as i64
}

impl CudaTags {
    fn insert(&mut self, pts: i64, tag: u64) -> Result<(), TierError> {
        if self.pending.len() >= 64
            || self.pending.contains_key(&pts)
            || self.pending.values().any(|value| *value == tag)
            || self.displayed.is_some_and(|last| pts <= last)
        {
            return Err(TierError::Codec(
                "CUDA presentation timestamp/tag collision, late input, or pending capacity exceeded; reset at an IDR"
                    .into(),
            ));
        }
        self.pending.insert(pts, tag);
        Ok(())
    }

    fn take(&mut self, pts: i64) -> Result<u64, TierError> {
        if self.displayed.is_some_and(|last| pts <= last) {
            return Err(TierError::Codec("CUDA returned nonmonotone presentation timestamps".into()));
        }
        let tag = self.pending.remove(&pts).ok_or_else(|| {
            TierError::Codec(
                "CUDA returned an unknown presentation timestamp; refusing to guess the original tag".into(),
            )
        })?;
        // Output is strictly in presentation order: lower pending timestamps can no longer be displayed.
        // Native H.264 may drop one while discovering reorder depth absent from the SPS. Report its original
        // tag so callers release retained headers and represent the loss rather than accumulating it forever.
        while self.pending.first_key_value().is_some_and(|(&pending, _)| pending < pts) {
            if let Some((_, dropped)) = self.pending.pop_first() {
                self.dropped.push(dropped);
            }
        }
        self.displayed = Some(pts);
        Ok(tag)
    }
}

// FFmpeg's documented hw_decode path: the codec owns the device reference, get_format refuses CPU fallback,
// and each returned GPU frame is downloaded once. No global device/format state or callback userdata is needed.
// https://ffmpeg.org/doxygen/trunk/hw_decode_8c-example.html
//
// Every decoder on a GPU retains that GPU's primary CUDA context (FFmpeg's `primary_ctx`) instead of creating a
// context of its own. Each CUDA context runs a driver event thread that wakes about 300 to 450 times a second even
// when idle, so see-decode's six cameras had six of them, more than half its wakeups (#3235); one shared context has
// one. The primary context is the one the CUDA runtime (torch, ONNX Runtime) uses, and FFmpeg refuses it when it is
// already active with scheduling flags other than its blocking sync, so a decoder then gets a private context as
// before.
fn cuda_context(context: &mut ff::codec::context::Context, gpu: u32) -> Result<(), TierError> {
    let device = std::ffi::CString::new(gpu.to_string())
        .map_err(|_| TierError::Codec("CUDA decoder GPU ordinal contains a NUL".into()))?;
    let create = |options: *mut ff::ffi::AVDictionary| {
        let mut hardware = std::ptr::null_mut();
        // SAFETY: hardware points to an initialized null AVBufferRef output; device is a live NUL-terminated
        // ordinal, and options is null or a live dictionary FFmpeg only reads. On success the caller owns the one
        // reference; on failure FFmpeg leaves no reference to release.
        let result = unsafe {
            ff::ffi::av_hwdevice_ctx_create(
                &mut hardware,
                ff::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA,
                device.as_ptr(),
                options,
                0,
            )
        };
        (result, hardware)
    };
    let mut shared = ff::Dictionary::new();
    shared.set("primary_ctx", "1");
    // SAFETY: disown hands over the live dictionary, which own takes back (and frees) right after the call.
    let (mut result, mut hardware) = unsafe {
        let options = shared.disown();
        let created = create(options);
        drop(ff::Dictionary::own(options));
        created
    };
    if result < 0 {
        (result, hardware) = create(std::ptr::null_mut());
    }
    if result < 0 {
        return Err(decode_err("CUDA device", ff::Error::from(result)));
    }
    // SAFETY: the context is unopened and uniquely borrowed, and owns no previous hardware reference; the one
    // reference transfers to it, and its FFmpeg destructor releases it.
    // FFmpeg invokes get_format with a valid AV_PIX_FMT_NONE-terminated list for the duration of the call.
    unsafe {
        debug_assert!((*context.as_ptr()).hw_device_ctx.is_null());
        (*context.as_mut_ptr()).hw_device_ctx = hardware;
        (*context.as_mut_ptr()).get_format = Some(cuda_format);
    }
    Ok(())
}

unsafe extern "C" fn cuda_format(
    _context: *mut ff::ffi::AVCodecContext,
    mut formats: *const ff::ffi::AVPixelFormat,
) -> ff::ffi::AVPixelFormat {
    // SAFETY: FFmpeg's get_format contract guarantees a readable sentinel-terminated array. Only values from
    // that array are returned; NONE explicitly fails decoding when CUDA isn't offered, with no software fallback.
    unsafe {
        while *formats != ff::ffi::AVPixelFormat::AV_PIX_FMT_NONE {
            if *formats == ff::ffi::AVPixelFormat::AV_PIX_FMT_CUDA {
                return *formats;
            }
            formats = formats.add(1);
        }
    }
    ff::ffi::AVPixelFormat::AV_PIX_FMT_NONE
}

fn cuda_download(picture: &ff::frame::Video) -> Result<ff::frame::Video, TierError> {
    if picture.format() != Pixel::CUDA {
        return Err(TierError::Codec(
            "CUDA H.264 decoder returned a non-hardware frame; refusing software fallback".into(),
        ));
    }
    let mut host = ff::frame::Video::empty();
    // SAFETY: both frames are live, distinct AVFrames. The source owns its CUDA surface; the empty destination
    // receives FFmpeg-owned host buffers. Both wrappers release their buffers on every return path.
    let result = unsafe { ff::ffi::av_hwframe_transfer_data(host.as_mut_ptr(), picture.as_ptr(), 0) };
    if result < 0 {
        return Err(decode_err("CUDA frame transfer", ff::Error::from(result)));
    }
    // SAFETY: both AVFrames remain live and distinct. Copying reference-counted properties preserves PTS and
    // color signaling; av_hwframe_transfer_data copies the pixels and dimensions but does not copy properties.
    let result = unsafe { ff::ffi::av_frame_copy_props(host.as_mut_ptr(), picture.as_ptr()) };
    if result < 0 {
        return Err(decode_err("CUDA frame properties", ff::Error::from(result)));
    }
    // SAFETY: host owns valid CPU pixel buffers and the crop metadata just copied from the decoded frame.
    // FFmpeg checks the offsets, adjusts pointers/dimensions within those buffers, and clears applied offsets.
    let result = unsafe { ff::ffi::av_frame_apply_cropping(host.as_mut_ptr(), 0) };
    if result < 0 {
        return Err(decode_err("CUDA frame cropping", ff::Error::from(result)));
    }
    Ok(host)
}

/// One H.264 stream's decoder. Every access unit is fed in decode order before consumers sample images.
/// Reset after a packet/sequence gap or a new capture epoch, then wait for an actual IDR rather than decoding with
/// missing reference pictures. No subprocess, camera connection, or persistent decoded stream is involved.
pub struct H264StreamDecoder {
    backend: H264DecodeBackend,
    cuda_tags: CudaTags,
    decoder: ff::decoder::Video,
    waiting_for_keyframe: bool,
    /// AV1 temporal units (the archive pass's chunks, #1479) rather than H.264: software only.
    av1: bool,
    /// Tags of inputs pushed with `keep` false: their pictures decode (they are references) but are never downloaded
    /// or copied. Bounded by `SKIPPED_HELD`, oldest tags first, since a picture the codec drops never returns.
    skipped: std::collections::BTreeSet<u64>,
}

/// How many skipped inputs' tags a decoder remembers while their pictures are in flight.
const SKIPPED_HELD: usize = 64;

impl H264StreamDecoder {
    /// Opens a single-threaded CPU H.264 decoder, with a bounded picture allocation.
    /// # Errors
    /// Decoder initialization or opening errors.
    pub fn new() -> Result<Self, TierError> {
        Self::with_backend(H264DecodeBackend::Software)
    }

    /// Opens the selected backend explicitly, with one codec thread and a 16-million-pixel limit.
    /// CUDA uses FFmpeg's native H.264 parser and an NVDEC hardware context, then downloads host NV12.
    /// Unlike the CUVID parser, this does not hold 19 pictures on streams with incomplete VUI (#2080).
    /// Normal H.264 presentation reordering stays enabled; no LOW_DELAY flag or bitstream rewrite is used.
    /// No decoder/driver/device failure silently selects another backend.
    /// # Errors
    /// Invalid ordinal, unavailable decoder/device/driver, or codec initialization failure.
    pub fn with_backend(backend: H264DecodeBackend) -> Result<Self, TierError> {
        Self::open(backend, false)
    }

    /// As [`H264StreamDecoder::new`], for the AV1 temporal units of an archived chunk (#1479): one unit per
    /// [`push`](Self::push), starting at one that carries a sequence header ([`crate::gop::av1_key`]).
    /// # Errors
    /// Decoder initialization or opening errors, including an FFmpeg without an AV1 decoder.
    pub fn av1() -> Result<Self, TierError> {
        Self::open(H264DecodeBackend::Software, true)
    }

    fn open(backend: H264DecodeBackend, av1: bool) -> Result<Self, TierError> {
        if matches!(backend, H264DecodeBackend::Cuda { gpu } if gpu > i32::MAX as u32) {
            return Err(TierError::Codec("CUDA decoder GPU ordinal must fit a nonnegative i32".into()));
        }
        let _open = crate::encode::codec_open().map_err(|error| decode_err("init", error))?;
        let codec = if av1 {
            ff::decoder::find(ff::codec::Id::AV1)
                .ok_or_else(|| TierError::Codec("no AV1 decoder in this FFmpeg build".into()))?
        } else {
            ff::decoder::find(ff::codec::Id::H264)
                .ok_or_else(|| TierError::Codec("no native H.264 decoder in this FFmpeg build".into()))?
        };
        let mut context = ff::codec::context::Context::new_with_codec(codec);
        context.set_threading(ff::codec::threading::Config::count(1));
        let mut options = ff::Dictionary::new();
        options.set("max_pixels", "16777216");
        if let H264DecodeBackend::Cuda { gpu } = backend {
            cuda_context(&mut context, gpu)?;
        }
        let mut decoder_context = context.decoder();
        if matches!(backend, H264DecodeBackend::Cuda { .. }) {
            decoder_context.set_packet_time_base((1, 10_000_000));
        }
        let decoder = decoder_context
            .open_as_with(codec, options)
            .and_then(|opened| opened.video())
            .map_err(|error| TierError::Codec(format!("H.264 {backend:?} decoder could not open: {error}; check FFmpeg decoder support and the selected device/driver")))?;
        Ok(Self {
            backend,
            cuda_tags: CudaTags::default(),
            decoder,
            waiting_for_keyframe: true,
            av1,
            skipped: std::collections::BTreeSet::new(),
        })
    }

    /// Backend retained across resets. Hardware requests never silently change this value.
    #[must_use]
    pub const fn backend(&self) -> H264DecodeBackend {
        self.backend
    }

    /// Whether a reset/first-start is still waiting for an independently decodable picture.
    #[must_use]
    pub const fn waiting_for_keyframe(&self) -> bool {
        self.waiting_for_keyframe
    }

    /// Original input tags discarded by the codec during the most recent push or finish call. A caller retaining
    /// source headers must release these and report a gap. The list is bounded by the 64 pending inputs and is
    /// replaced on each call, so a caller that does not inspect it cannot accumulate reports indefinitely.
    #[must_use]
    pub fn dropped_tags(&self) -> &[u64] {
        &self.cuda_tags.dropped
    }

    /// Forgets reference pictures from an old capture epoch or a packet gap.
    /// # Errors
    /// Decoder reopening errors.
    pub fn reset(&mut self) -> Result<(), TierError> {
        *self = Self::open(self.backend, self.av1)?;
        Ok(())
    }

    /// Feeds one complete Annex-B access unit. Output may be empty or may name an older tag because of reordering.
    /// The caller retains each original header by tag until its decoded picture returns.
    /// # Errors
    /// Oversized/empty input, an invalid tag, damaged bitstream, unsupported pixel format, or decoder failure.
    pub fn push(&mut self, bytes: &[u8], tag: u64) -> Result<Vec<TaggedFrame>, TierError> {
        if matches!(self.backend, H264DecodeBackend::Cuda { .. }) {
            return Err(TierError::Codec(
                "CUDA decoding requires push_with_pts with the actual presentation timestamp; opaque tags are not PTS"
                    .into(),
            ));
        }
        self.push_inner(bytes, tag, None, false, true)
    }

    /// Feed an access unit with its actual presentation timestamp, in nanoseconds. CUDA's
    /// packet clock retains 100 ns resolution: timestamps are rounded to its nearest tick, and collisions or
    /// nonmonotone displayed timestamps are rejected rather than assigning guessed original tags.
    /// Software retains the original opaque-tag behavior and does not use `presentation_ns`.
    /// # Errors
    /// Invalid input, timestamp collision, bounded pending-tag overflow, or decoder failure.
    pub fn push_with_pts(
        &mut self,
        bytes: &[u8],
        tag: u64,
        presentation_ns: i64,
    ) -> Result<Vec<TaggedFrame>, TierError> {
        self.push_inner(bytes, tag, Some(presentation_ns), false, true)
    }

    /// Decode once, returning actual color metadata and optionally signaled-color BGR.
    /// Unsupported or unspecified colorimetry returns no BGR; NV12 remains available.
    /// With `keep` false the unit still decodes (later pictures may reference it), but its picture is never
    /// downloaded from the GPU or copied: it returns with its tag, empty `nv12`, and no BGR (#3218).
    /// # Errors
    /// The same input/identity errors as push, plus supported-color conversion failure.
    pub fn push_outputs(
        &mut self,
        bytes: &[u8],
        tag: u64,
        presentation_ns: Option<i64>,
        want_bgr: bool,
        keep: bool,
    ) -> Result<Vec<TaggedFrame>, TierError> {
        if matches!(self.backend, H264DecodeBackend::Cuda { .. }) && presentation_ns.is_none() {
            return Err(TierError::Codec("CUDA decoding requires an actual presentation timestamp".into()));
        }
        self.push_inner(bytes, tag, presentation_ns, want_bgr, keep)
    }

    fn push_inner(
        &mut self,
        bytes: &[u8],
        tag: u64,
        presentation_ns: Option<i64>,
        want_bgr: bool,
        keep: bool,
    ) -> Result<Vec<TaggedFrame>, TierError> {
        self.cuda_tags.dropped.clear();
        if bytes.is_empty() || bytes.len() > 16 * 1024 * 1024 {
            return Err(TierError::Codec("H.264 access unit must contain at most 16 MiB".into()));
        }
        let cuda = matches!(self.backend, H264DecodeBackend::Cuda { .. });
        let timestamp = if cuda {
            cuda_tick(presentation_ns.ok_or_else(|| TierError::Codec("CUDA presentation timestamp is missing".into()))?)
        } else {
            i64::try_from(tag).map_err(|_| TierError::Codec("H.264 input tag exceeds i64".into()))?
        };
        if self.waiting_for_keyframe {
            let key = if self.av1 { crate::gop::av1_key(bytes) } else { crate::gop::h264_idr(bytes) };
            if !key {
                return Ok(Vec::new());
            }
            self.waiting_for_keyframe = false;
        }
        if cuda {
            self.cuda_tags.insert(timestamp, tag)?;
        }
        if !keep {
            self.skipped.insert(tag);
            while self.skipped.len() > SKIPPED_HELD {
                self.skipped.pop_first();
            }
        }
        let mut packet = ff::Packet::copy(bytes);
        packet.set_pts(Some(timestamp));
        if let Err(error) = self.decoder.send_packet(&packet) {
            if cuda {
                self.cuda_tags.pending.remove(&timestamp);
            }
            self.skipped.remove(&tag);
            return Err(decode_err("send", error));
        }
        self.take(want_bgr)
    }

    /// Drains delayed pictures only at end of stream, never between live access units.
    /// # Errors
    /// Decoder drain errors.
    pub fn finish(&mut self) -> Result<Vec<TaggedFrame>, TierError> {
        self.finish_outputs(false)
    }

    /// Drain delayed pictures with the same optional color-safe BGR conversion as push_outputs.
    /// # Errors
    /// Decoder drain, identity, or supported-color conversion errors.
    pub fn finish_outputs(&mut self, want_bgr: bool) -> Result<Vec<TaggedFrame>, TierError> {
        self.cuda_tags.dropped.clear();
        self.decoder.send_eof().map_err(|error| decode_err("eof", error))?;
        let output = self.take(want_bgr)?;
        if !self.cuda_tags.pending.is_empty() {
            return Err(TierError::Codec("CUDA decoder ended without returning every original timestamp tag".into()));
        }
        Ok(output)
    }

    fn take(&mut self, want_bgr: bool) -> Result<Vec<TaggedFrame>, TierError> {
        let mut output = Vec::new();
        let mut picture = ff::frame::Video::empty();
        loop {
            match self.decoder.receive_frame(&mut picture) {
                Ok(()) => {}
                Err(ff::Error::Other { errno: ff::util::error::EAGAIN } | ff::Error::Eof) => break,
                Err(error) => return Err(decode_err("frame", error)),
            }
            let timestamp = picture
                .pts()
                .ok_or_else(|| TierError::Codec("H.264 decoder returned a picture without its timestamp".into()))?;
            let tag = if matches!(self.backend, H264DecodeBackend::Cuda { .. }) {
                self.cuda_tags.take(timestamp)?
            } else {
                u64::try_from(timestamp)
                    .map_err(|_| TierError::Codec("H.264 decoder returned an invalid input tag".into()))?
            };
            for dropped in &self.cuda_tags.dropped {
                self.skipped.remove(dropped);
            }
            if self.skipped.remove(&tag) {
                // Nobody takes this picture: no download, no copy. Its size and color are the decoded surface's.
                let (color_range, color_matrix, _) = live_color_outputs(&picture, false)?;
                output.push(TaggedFrame {
                    color_range,
                    color_matrix,
                    bgr: None,
                    tag,
                    width: picture.width(),
                    height: picture.height(),
                    nv12: Vec::new(),
                });
                continue;
            }
            if matches!(self.backend, H264DecodeBackend::Cuda { .. }) {
                picture = cuda_download(&picture)?;
            }
            normalize_decoded_range(&mut picture);
            if picture.width() == 0
                || picture.height() == 0
                || !picture.width().is_multiple_of(2)
                || !picture.height().is_multiple_of(2)
                || u64::from(picture.width()) * u64::from(picture.height()) > 16_777_216
            {
                return Err(TierError::Codec("H.264 decoded dimensions exceed NV12 bounds".into()));
            }
            let (color_range, color_matrix, bgr) = live_color_outputs(&picture, want_bgr)?;
            output.push(TaggedFrame {
                color_range,
                color_matrix,
                bgr,
                tag,
                width: picture.width(),
                height: picture.height(),
                nv12: to_nv12(&picture)?,
            });
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_outputs_report_color_and_convert_bgr_only_for_supported_explicit_signals() {
        use ff::util::color::{Range, Space};
        drop(crate::encode::codec_open().unwrap());
        let mut frame = ff::frame::Video::new(Pixel::NV12, 4, 2);
        for row in 0..2 {
            let stride = frame.stride(0);
            frame.data_mut(0)[row * stride..row * stride + 4].fill(100);
        }
        frame.data_mut(1)[..4].copy_from_slice(&[90, 200, 90, 200]);
        for range in [Range::JPEG, Range::MPEG] {
            frame.set_color_range(range);
            for (matrix, label) in
                [(Space::SMPTE170M, "bt601"), (Space::BT709, "bt709"), (Space::BT2020NCL, "bt2020nc")]
            {
                frame.set_color_space(matrix);
                let (actual_range, actual_matrix, bgr) = live_color_outputs(&frame, true).unwrap();
                assert_eq!(actual_range, if range == Range::JPEG { "full" } else { "limited" });
                assert_eq!(actual_matrix, label);
                let mut expected = signaled_rgb(&frame).unwrap().rgb;
                for pixel in expected.chunks_exact_mut(3) {
                    pixel.swap(0, 2);
                }
                assert_eq!(bgr, Some(expected));
                assert!(live_color_outputs(&frame, false).unwrap().2.is_none());
            }
        }
        frame.set_color_space(Space::Unspecified);
        assert_eq!(live_color_outputs(&frame, true).unwrap(), ("limited", "unspecified", None));
        frame.set_color_space(Space::BT2020CL);
        assert!(live_color_outputs(&frame, true).unwrap().2.is_none());
        frame.set_color_space(Space::BT709);
        frame.set_color_range(Range::Unspecified);
        assert_eq!(live_color_outputs(&frame, true).unwrap(), ("unspecified", "bt709", None));
    }

    #[test]
    fn rgb_conversion_honors_signaled_range_and_matrix_and_refuses_unknown() {
        use ff::util::color::{Range, Space};
        drop(crate::encode::codec_open().unwrap());
        let mut frame = ff::frame::Video::new(Pixel::NV12, 4, 2);
        for row in 0..2 {
            let stride = frame.stride(0);
            frame.data_mut(0)[row * stride..row * stride + 4].fill(100);
        }
        frame.data_mut(1)[..4].copy_from_slice(&[90, 200, 90, 200]);
        frame.set_color_range(Range::JPEG);
        for (matrix, expected) in [(Space::SMPTE170M, [201u8, 62, 33]), (Space::BT709, [213u8, 73, 29])] {
            frame.set_color_space(matrix);
            let rgb = signaled_rgb(&frame).unwrap();
            for pixel in rgb.rgb.chunks_exact(3) {
                assert!(
                    pixel.iter().zip(expected).all(|(&got, want)| got.abs_diff(want) <= 3),
                    "{matrix:?}: {pixel:?}"
                );
            }
        }
        frame.set_color_space(Space::Unspecified);
        assert!(signaled_rgb(&frame).is_err(), "an unknown matrix must not silently become BT.601");
        frame.set_color_space(Space::SMPTE170M);
        frame.set_color_range(Range::Unspecified);
        assert!(signaled_rgb(&frame).is_err(), "an unknown range must not silently become limited");
    }
}

#[cfg(test)]
mod cuda_timestamp_tests {
    use super::{CudaTags, cuda_tick};

    #[test]
    fn native_clock_rounding_and_collisions_never_guess_original_tags() {
        assert_eq!(cuda_tick(49), 0);
        assert_eq!(cuda_tick(50), 1);
        assert_eq!(cuda_tick(-50), -1);
        assert_eq!(cuda_tick(i64::MAX), 92_233_720_368_547_758);
        let mut tags = CudaTags::default();
        tags.insert(cuda_tick(10_000), 70).unwrap();
        assert!(tags.insert(cuda_tick(10_001), 71).is_err());
        tags.insert(cuda_tick(30_000), 72).unwrap();
        tags.insert(cuda_tick(20_000), 71).unwrap();
        assert_eq!(tags.take(cuda_tick(10_000)).unwrap(), 70);
        assert_eq!(tags.take(cuda_tick(20_000)).unwrap(), 71);
        assert_eq!(tags.take(cuda_tick(30_000)).unwrap(), 72);
        assert!(tags.take(cuda_tick(40_000)).is_err());
        assert!(tags.insert(cuda_tick(20_000), 73).is_err());
    }

    #[test]
    fn a_dropped_cuda_picture_does_not_accumulate_pending_original_tags() {
        let mut tags = CudaTags::default();
        tags.insert(0, 10).unwrap();
        tags.insert(1, 11).unwrap();
        tags.insert(2, 12).unwrap();
        assert_eq!(tags.take(0).unwrap(), 10);
        // The codec discarded picture 1 while discovering an initially unspecified reorder depth.
        assert_eq!(tags.take(2).unwrap(), 12);
        assert!(!tags.pending.contains_key(&1), "a skipped original tag must not stay pending forever");
        assert_eq!(tags.dropped, [11], "the caller needs the discarded picture's original input tag");
        for pts in 3..128 {
            tags.dropped.clear(); // a new decoder call replaces the previous call's loss report
            tags.insert(2 * pts, pts as u64 + 1000).unwrap();
            tags.insert(2 * pts + 1, pts as u64).unwrap();
            assert_eq!(tags.take(2 * pts + 1).unwrap(), pts as u64);
            assert_eq!(tags.dropped, [pts as u64 + 1000]);
        }
        assert!(tags.pending.is_empty());
    }

    #[test]
    fn pending_original_tags_are_bounded_and_duplicate_tags_are_rejected() {
        let mut tags = CudaTags::default();
        for n in 0..64 {
            tags.insert(n, n as u64).unwrap();
        }
        assert!(tags.insert(64, 64).is_err());
        assert_eq!(tags.take(0).unwrap(), 0);
        assert!(tags.insert(65, 1).is_err());
        tags.insert(64, 64).unwrap();
        assert_eq!(tags.pending.len(), 64);
    }
}
