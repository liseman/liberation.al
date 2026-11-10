//! Exporting a time range of a video stream as an ordinary MP4, behind the `ffmpeg` feature.
//!
//! Chunks are the tier worker's own container, which no video player or ffprobe opens (docs/CONTINUUM-STORAGE.md
//! says why). [`export`] writes a range of one stream's chunks as an MP4 that does open, for when a person
//! wants a clip in a player. The `continuum-export` command is the way to run it.
//!
//! - [`ExportCodec::Copy`]: a raw camera's H.264 access units ([`ChunkFormat::H264Au`]) are muxed as they are, with
//!   no re-encode. A copy can only start on a keyframe, so it starts at the keyframe before the range.
//! - [`ExportCodec::H264`]: the frames are decoded (JPEG for an MJPEG chunk, the GOPs for H.264) and encoded as H.264
//!   ([`crate::h264`]), which every player decodes; the range is exact.
//! - [`ExportCodec::Auto`]: copy a raw camera's H.264, or re-encode MJPEG chunks, the archive pass's AV1 chunks
//!   ([`ChunkFormat::Av1Au`], so the clip plays everywhere), or mixed dimensions.
//!
//! Re-encoding uses the largest selected source frame by pixel area (then width and height for ties) as its
//! fixed canvas. Other frames scale to fit with their aspect ratio preserved to the nearest even pixel and
//! centered black bars. Native chunks remain untouched. Copy rejects dimension changes with H.264 guidance.
//!
//! The MP4 is written with `+faststart` off and no fragmenting: it's a finished file, not something a crash could
//! interrupt halfway (an interrupted export just leaves a file to delete).

use std::path::Path;

use ff::codec::Id;
use ff::{Packet, Rational};
use ffmpeg_next as ff;

use crate::TierError;
use crate::chunk::{ChunkFormat, ChunkReader};
use crate::encode::AccessUnit;
use crate::h264::{H264Encoder, H264Settings};
use crate::hydrate::{decode_jpeg, rgb_to_nv12};
use crate::video::{DecodedFrame, decode_gop};

/// FFmpeg's `AV_INPUT_BUFFER_PADDING_SIZE`: extradata must be allocated this much longer, zeroed.
const PADDING: usize = 64;

/// How an export writes its video.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportCodec {
    /// Copy a raw camera's H.264 GOPs, otherwise encode H.264 with a single aspect-preserving canvas.
    Auto,
    /// Mux GOPs as they are, rounded out to whole GOPs. Only for a raw camera's Tier 1 ([`ChunkFormat::H264Au`]):
    /// it starts at the keyframe before the range and ends with it.
    Copy,
    /// Decode and re-encode as H.264, exactly the range. Works for every video chunk.
    H264,
}

/// What an export wrote.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Exported {
    /// Frames in the file.
    pub frames: usize,
    /// Width and height in pixels.
    pub size: (u32, u32),
}

fn err(what: &str, e: ff::Error) -> TierError {
    TierError::Codec(format!("export {what}: {e}"))
}

fn overlaps(start_ns: u64, end_ns: u64, from: u64, to: u64) -> bool {
    start_ns < to && end_ns.max(start_ns.saturating_add(1)) > from
}

/// Where an export's access units go as they're made, with the frame size: the muxer, so nothing holds the whole
/// range in memory.
type Sink<'a> = dyn FnMut(AccessUnit, (u32, u32)) -> Result<(), TierError> + 'a;

/// Whether every chunk can be copied: a raw camera's H.264 access units.
///
/// # Errors
///
/// [`TierError::Codec`] for a chunk that isn't [`ChunkFormat::H264Au`].
fn copyable(chunks: &[&Path]) -> Result<(), TierError> {
    for &path in chunks {
        if ChunkReader::open(path)?.format() != ChunkFormat::H264Au {
            return Err(TierError::Codec(format!("{} isn't H.264 video: export it as H.264", path.display())));
        }
    }
    Ok(())
}

/// Sends the units of every GOP in `chunks` that overlaps `[from, to)` to `sink`, in order, one GOP at a time
/// ([`crate::tier2::for_each_gop`]: a raw camera's run of access units from a keyframe).
fn whole_gops(chunks: &[&Path], from: u64, to: u64, sink: &mut Sink<'_>) -> Result<(), TierError> {
    let mut size = None;
    for &path in chunks {
        let mut reader = ChunkReader::open(path)?;
        crate::tier2::for_each_gop(&mut reader, from, to, &mut |gop| {
            for frame in decode_gop(&gop)? {
                let next = (frame.width, frame.height);
                if size.is_some_and(|previous| previous != next) {
                    return Err(TierError::Codec(
                        "copy crosses frame sizes: export as H.264 (--codec h264)".to_owned(),
                    ));
                }
                size = Some(next);
            }
            for u in gop {
                let at = AccessUnit { bytes: u.bytes, start_ns: u.start_ns, end_ns: u.end_ns, key: u.key };
                sink(at, size.unwrap_or((0, 0)))?;
            }
            Ok(())
        })?;
    }
    Ok(())
}

/// Visits selected native frames, holding at most one GOP's decoded frames at a time.
fn frames(
    chunks: &[&Path],
    from: u64,
    to: u64,
    each: &mut dyn FnMut(DecodedFrame) -> Result<(), TierError>,
) -> Result<(), TierError> {
    for &path in chunks {
        let mut reader = ChunkReader::open(path)?;
        match reader.format() {
            ChunkFormat::Passthrough => {
                for (index, record) in reader.records_in(from, to) {
                    if record.cut {
                        continue;
                    }
                    let picture = decode_jpeg(&reader.read(index)?)?;
                    let size = (picture.width, picture.height);
                    each(DecodedFrame {
                        width: size.0,
                        height: size.1,
                        start_ns: record.start_ns,
                        end_ns: record.end_ns,
                        nv12: rgb_to_nv12(&picture),
                    })?;
                }
            }
            ChunkFormat::H264Au | ChunkFormat::Av1Au => {
                drop(reader);
                crate::tier2::for_each_frame(path, from, to, each)?;
            }
            other => return Err(TierError::Codec(format!("{} holds {other:?}, not video", path.display()))),
        }
    }
    Ok(())
}

/// Preflight only the requested frames. Largest means pixel area, with dimensions breaking ties, so input
/// order cannot choose the canvas. Frames remain bounded to a GOP rather than collecting the export in memory.
fn geometry(chunks: &[&Path], from: u64, to: u64) -> Result<((u32, u32), bool), TierError> {
    let mut largest: Option<(u64, (u32, u32))> = None;
    let mut first = None;
    let mut mixed = false;
    frames(chunks, from, to, &mut |f| {
        let size = (f.width, f.height);
        mixed |= first.is_some_and(|s| s != size);
        first.get_or_insert(size);
        let key = (u64::from(f.width) * u64::from(f.height), size);
        largest = Some(largest.map_or(key, |old| old.max(key)));
        Ok(())
    })?;
    largest.map(|(_, size)| (size, mixed)).ok_or_else(|| TierError::Codec("nothing in that range to export".to_owned()))
}

/// Scale NV12 into an even-sized centered rectangle, then pad it with full-range black. Chroma offsets and
/// dimensions are even because NV12 shares one UV pair between four luma samples.
fn fit(frame: &DecodedFrame, target: (u32, u32)) -> Result<Vec<u8>, TierError> {
    use ff::format::Pixel;
    use ff::software::scaling::{Context, Flags};

    if (frame.width, frame.height) == target {
        return Ok(frame.nv12.clone());
    }
    let (w, h) = (target.0 as usize, target.1 as usize);
    let (sw, sh) = (frame.width as usize, frame.height as usize);
    let (fw, fh) = if w * sh <= h * sw { (w, (sh * w / sw).max(2) & !1) } else { ((sw * h / sh).max(2) & !1, h) };
    let mut input = ff::frame::Video::new(Pixel::NV12, frame.width, frame.height);
    for (plane, rows, offset) in [(0, sh, 0), (1, sh / 2, sw * sh)] {
        let stride = input.stride(plane);
        for row in 0..rows {
            input.data_mut(plane)[row * stride..row * stride + sw]
                .copy_from_slice(&frame.nv12[offset + row * sw..offset + (row + 1) * sw]);
        }
    }
    let mut scaler =
        Context::get(Pixel::NV12, frame.width, frame.height, Pixel::NV12, fw as u32, fh as u32, Flags::BILINEAR)
            .map_err(|e| err("scale", e))?;
    let mut scaled = ff::frame::Video::empty();
    scaler.run(&input, &mut scaled).map_err(|e| err("scale", e))?;
    let mut out = vec![0; w * h * 3 / 2];
    out[w * h..].fill(128);
    let (x, y) = (((w - fw) / 2) & !1, ((h - fh) / 2) & !1);
    for (plane, rows, offset, top) in [(0, fh, 0, y), (1, fh / 2, w * h, y / 2)] {
        let stride = scaled.stride(plane);
        for row in 0..rows {
            let dest = offset + (top + row) * w + x;
            out[dest..dest + fw].copy_from_slice(&scaled.data(plane)[row * stride..row * stride + fw]);
        }
    }
    Ok(out)
}

fn reencode(chunks: &[&Path], from: u64, to: u64, size: (u32, u32), sink: &mut Sink<'_>) -> Result<(), TierError> {
    let mut encoder = H264Encoder::open(size.0, size.1, H264Settings::default())?;
    frames(chunks, from, to, &mut |f| {
        for au in encoder.encode(&fit(&f, size)?, f.start_ns, f.end_ns)? {
            sink(au, size)?;
        }
        Ok(())
    })?;
    for au in encoder.finish()? {
        sink(au, size)?;
    }
    Ok(())
}

/// The H.264 parameter-set NAL units (SPS and PPS) of an Annex B access unit, each with its start code: the stream's
/// extradata.
fn parameter_sets(au: &[u8]) -> Vec<u8> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= au.len() {
        if au[i..i + 3] == [0, 0, 1] {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::new();
    for (k, &s) in starts.iter().enumerate() {
        let end = starts.get(k + 1).map_or(au.len(), |&next| {
            // Drop the next unit's start code, including a four-byte one's leading zero.
            let mut e = next - 3;
            if e > s && au[e - 1] == 0 {
                e -= 1;
            }
            e
        });
        let Some(&head) = au.get(s) else { continue };
        if matches!(head & 0x1f, 7 | 8) {
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(&au[s..end]);
        }
    }
    out
}

/// An MP4 being written one access unit at a time. It opens the file on the first unit, whose parameter sets are
/// the stream's extradata.
struct Muxer {
    path: std::path::PathBuf,
    open: Option<Open>,
    frames: usize,
    size: (u32, u32),
}

struct Open {
    octx: ff::format::context::Output,
    time_base: Rational,
    t0: u64,
    last_start: u64,
}

impl Muxer {
    fn new(path: &Path) -> Self {
        Self { path: path.to_path_buf(), open: None, frames: 0, size: (0, 0) }
    }

    fn push(&mut self, au: &AccessUnit, size: (u32, u32)) -> Result<(), TierError> {
        let us = Rational(1, 1_000_000);
        let open = match &mut self.open {
            Some(o) => {
                // One stream's units in decode order, which (no B-frames) is time order: anything else is chunks
                // out of order, and would also make the times below run backwards.
                if au.start_ns < o.last_start {
                    return Err(TierError::Codec(
                        "the frames aren't in time order: pass one stream's chunks, in order".to_owned(),
                    ));
                }
                o
            }
            None => {
                if !au.key {
                    return Err(TierError::Codec("an export must start on a keyframe".to_owned()));
                }
                self.size = size;
                self.open.insert(open_mp4(&self.path, size, &au.bytes, au.start_ns)?)
            }
        };
        let mut packet = Packet::copy(&au.bytes);
        let pts = ((au.start_ns - open.t0) / 1_000) as i64;
        packet.set_pts(Some(pts));
        packet.set_dts(Some(pts));
        // A captured instant can span only one nanosecond. Rounding that down makes MP4 mark
        // a lone packet as discardable, so a retained face frame decodes as an empty movie.
        packet.set_duration((au.end_ns.saturating_sub(au.start_ns).div_ceil(1_000).max(1)) as i64);
        if au.key {
            packet.set_flags(ff::codec::packet::Flags::KEY);
        }
        packet.set_stream(0);
        packet.rescale_ts(us, open.time_base);
        packet.write_interleaved(&mut open.octx).map_err(|e| err("write", e))?;
        open.last_start = au.start_ns;
        self.frames += 1;
        Ok(())
    }

    /// Writes the trailer: how many frames, and their size.
    fn finish(self) -> Result<(usize, (u32, u32)), TierError> {
        let Some(mut open) = self.open else {
            return Err(TierError::Codec("nothing in that range to export".to_owned()));
        };
        open.octx.write_trailer().map_err(|e| err("trailer", e))?;
        Ok((self.frames, self.size))
    }
}

/// Opens the MP4 with one video stream, its extradata the parameter sets of `first` (a keyframe).
fn open_mp4(out: &Path, size: (u32, u32), first: &[u8], t0: u64) -> Result<Open, TierError> {
    let id = Id::H264;
    let extradata = parameter_sets(first);
    if extradata.is_empty() {
        return Err(TierError::Codec("the first keyframe has no parameter sets".to_owned()));
    }
    let mut octx = ff::format::output_as(out, "mp4").map_err(|e| err("open", e))?;
    let us = Rational(1, 1_000_000);
    {
        let mut stream = octx.add_stream(ff::encoder::find(id)).map_err(|e| err("stream", e))?;
        stream.set_time_base(us);
        // SAFETY: the stream was just created by avformat_new_stream and owns a valid codecpar, which nothing else
        // holds while this block writes it. The extradata buffer comes from av_mallocz with the padding FFmpeg
        // requires, so FFmpeg frees it with the stream.
        unsafe {
            let par = (*stream.as_mut_ptr()).codecpar;
            (*par).codec_type = ff::ffi::AVMediaType::AVMEDIA_TYPE_VIDEO;
            (*par).codec_id = id.into();
            (*par).width = size.0 as i32;
            (*par).height = size.1 as i32;
            let buf = ff::ffi::av_mallocz(extradata.len() + PADDING).cast::<u8>();
            if buf.is_null() {
                return Err(TierError::Codec("export: out of memory".to_owned()));
            }
            std::ptr::copy_nonoverlapping(extradata.as_ptr(), buf, extradata.len());
            (*par).extradata = buf;
            (*par).extradata_size = extradata.len() as i32;
        }
    }
    octx.write_header().map_err(|e| err("header", e))?;
    let time_base = octx.stream(0).map_or(us, |s| s.time_base());
    Ok(Open { octx, time_base, t0, last_start: t0 })
}

/// Writes the frames of `chunks` (one stream's, in time order) that fall in `[from, to)` to `out` as an MP4. The
/// file is written beside `out` and renamed into place when it's complete, replacing any file there, so a failure
/// leaves nothing at `out`.
///
/// # Errors
///
/// [`TierError::Codec`] for nothing in the range, chunks of more than one stream or out of time order,
/// [`ExportCodec::Copy`] of incompatible codecs or dimensions, a non-video chunk, or an
/// FFmpeg failure; chunk and I/O errors as they come.
pub fn export(chunks: &[&Path], out: &Path, from: u64, to: u64, codec: ExportCodec) -> Result<Exported, TierError> {
    if to <= from {
        return Err(TierError::Codec(format!("an empty or backwards range: [{from}, {to})")));
    }
    let mut streams = chunks.iter().map(|p| ChunkReader::open(p).map(|r| r.stream()));
    if let Some(first) = streams.next().transpose()? {
        for s in streams {
            if s? != first {
                return Err(TierError::Codec("the chunks are from more than one stream".to_owned()));
            }
        }
    }
    drop(crate::encode::codec_open().map_err(|e| err("init", e))?);
    let layout = if codec == ExportCodec::Copy { None } else { Some(geometry(chunks, from, to)?) };
    let codec = match codec {
        ExportCodec::Auto if !layout.is_some_and(|(_, mixed)| mixed) && copyable(chunks).is_ok() => ExportCodec::Copy,
        ExportCodec::Auto => ExportCodec::H264,
        other => other,
    };
    if codec == ExportCodec::Copy {
        copyable(chunks)?;
    }
    let temp = crate::chunk::partial(out);
    let mut muxer = Muxer::new(&temp);
    let written = {
        let mut sink = |au: AccessUnit, size: (u32, u32)| match codec {
            // A copy keeps whole GOPs; a re-encode keeps exactly the range.
            ExportCodec::H264 if !overlaps(au.start_ns, au.end_ns, from, to) => Ok(()),
            _ => muxer.push(&au, size),
        };
        match codec {
            ExportCodec::Copy => whole_gops(chunks, from, to, &mut sink),
            ExportCodec::H264 | ExportCodec::Auto => {
                reencode(chunks, from, to, layout.expect("reencode geometry").0, &mut sink)
            }
        }
    }
    .and_then(|()| muxer.finish());
    let (frames, size) = match written {
        Ok(w) => w,
        Err(e) => {
            let _ = std::fs::remove_file(&temp);
            return Err(e);
        }
    };
    std::fs::rename(&temp, out)?;
    Ok(Exported { frames, size })
}
