//! Reading frames back out of chunks (hydrate).
//!
//! A hydrate job decodes exactly the requested frames into Tier 0 slots. This module gives the bytes; the worker
//! copies them into the slots the broker leased and commits new payloads whose parent is the chunk, so anything
//! computed from them records `source_tier` as the tier they came from (docs/CONTINUUM.md, "Storage tiers").

use std::path::Path;

use crate::TierError;
use crate::chunk::{ChunkError, ChunkFormat, ChunkReader, Record};
use crate::codecs::lz4_decompress;

/// One frame read back from a chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hydrated {
    /// The frame's index in the chunk.
    pub index: usize,
    /// Its record.
    pub record: Record,
    /// Its bytes in the stream's native form: JPEG for an MJPEG stream, raw pixels for an LZ4 stream (the LZ4 is
    /// undone), a microphone's ring records exactly as they were for a PCM batch stream ([`crate::pcm`]), FLAC for a FLAC stream (decoding to PCM is the consumer's, or a decode-on-request capability's), and
    /// a unit record for a raw camera's H.264 ([`crate::gop::decode`] gives its access unit).
    pub bytes: Vec<u8>,
    /// Exact microphone format for CPB2 or Opus batches. None for legacy unknown-rate audio and nonaudio.
    pub pcm: Option<crate::pcm::PcmFormat>,
}

/// Reads the frames of `path` whose time overlaps `[start_ns, end_ns)`.
///
/// # Errors
///
/// A chunk error (not a chunk, corrupt record), or an I/O error.
pub fn hydrate(path: &Path, start_ns: u64, end_ns: u64) -> Result<Vec<Hydrated>, TierError> {
    if let Some(whole) = foreign_flac(path)? {
        return Ok(vec![whole]);
    }
    hydrate_reader(ChunkReader::open(path)?, start_ns, end_ns)
}

fn hydrate_reader(mut reader: ChunkReader, start_ns: u64, end_ns: u64) -> Result<Vec<Hydrated>, TierError> {
    let wanted = reader.records_in(start_ns, end_ns);
    let mut logical: Vec<_> = (0..reader.records().len()).collect();
    if matches!(reader.format(), ChunkFormat::H264Au | ChunkFormat::Av1Au) {
        for (catalog_index, physical) in crate::flush::catalog_record_indexes(reader.records()).into_iter().enumerate()
        {
            logical[physical] = catalog_index;
        }
    }
    let mut out = Vec::with_capacity(wanted.len());
    for (index, record) in wanted {
        let raw = reader.read(index)?;
        let pcm = pcm_format(reader.format(), &raw)?;
        let bytes = native(reader.format(), raw)?;
        out.push(Hydrated { index: logical[index], record, bytes, pcm });
    }
    Ok(out)
}

/// Reads one frame by its index in the chunk (what a `FRAMES` row points at).
///
/// # Errors
///
/// [`TierError::Purged`] for a frame a purge cut, a chunk error, or an I/O error.
pub fn hydrate_frame(path: &Path, index: usize) -> Result<Hydrated, TierError> {
    if let Some(whole) = foreign_flac(path)? {
        return if index == 0 { Ok(whole) } else { Err(crate::chunk::ChunkError::NoSuchFrame(index).into()) };
    }
    hydrate_reader_frame(ChunkReader::open(path)?, index)
}

fn hydrate_reader_frame(mut reader: ChunkReader, index: usize) -> Result<Hydrated, TierError> {
    let physical = if matches!(reader.format(), ChunkFormat::H264Au | ChunkFormat::Av1Au) {
        *crate::flush::catalog_record_indexes(reader.records())
            .get(index)
            .ok_or(crate::chunk::ChunkError::NoSuchFrame(index))?
    } else {
        index
    };
    let record = *reader.records().get(physical).ok_or(crate::chunk::ChunkError::NoSuchFrame(index))?;
    if record.cut {
        return Err(TierError::Purged(index));
    }
    let raw = reader.read(physical)?;
    let pcm = pcm_format(reader.format(), &raw)?;
    let bytes = native(reader.format(), raw)?;
    Ok(Hydrated { index, record, bytes, pcm })
}

fn pcm_format(format: ChunkFormat, raw: &[u8]) -> Result<Option<crate::pcm::PcmFormat>, TierError> {
    match format {
        ChunkFormat::PcmBatch => crate::pcm::format_of(raw),
        ChunkFormat::Opus => crate::opus::format_of(raw).map(Some),
        _ => Ok(None),
    }
}

/// A record's bytes in the stream's native form (see [`Hydrated::bytes`]).
fn native(format: ChunkFormat, raw: Vec<u8>) -> Result<Vec<u8>, TierError> {
    Ok(match format {
        ChunkFormat::Lz4 => lz4_decompress(&raw)?,
        ChunkFormat::PcmBatch => crate::pcm::decode_batch(&raw)?,
        ChunkFormat::Mp4 => return Err(mp4_in_a_chunk()),
        ChunkFormat::Opus => crate::opus::decode_batch(&raw)?,
        ChunkFormat::Passthrough
        | ChunkFormat::Flac
        | ChunkFormat::ArrowIpc
        | ChunkFormat::H264Au
        | ChunkFormat::Av1Au => raw,
    })
}

/// A chunk's live records whose time overlaps `[start_ns, end_ns)`, in the stream's native form, for a consumer
/// reading a flushed query item itself: a query hands back a chunk as one payload (its Chunk locator names the whole
/// file), and its records are the payloads it absorbed, each with its own id. Records a purge cut are left out. `None`
/// when the file is gone (a purge took the whole chunk), so the caller asks for the chunk again. The file is open
/// only during the call, so it never holds up a purge's rewrite (a new file renamed over the old) or unlink. A v1 FLAC
/// file registered in place is one record, the whole file, with zero times and the nil id (its query item's header
/// has its span and id).
///
/// # Errors
///
/// A corrupt record, a file that is not a chunk, or an I/O/codec error.
pub fn read_chunk(path: &Path, start_ns: u64, end_ns: u64) -> Result<Option<Vec<Hydrated>>, TierError> {
    let read = (|| {
        if let Some(whole) = foreign_flac(path)? {
            return Ok(vec![whole]);
        }
        hydrate_reader(ChunkReader::open(path)?, start_ns, end_ns)
    })();
    match read {
        Err(TierError::Chunk(ChunkError::Io(e)) | TierError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
            Ok(None)
        }
        other => other.map(Some),
    }
}

/// A v1 FLAC file registered in place ([`crate::v1audio`]): the whole file, as the one record of a chunk whose
/// times live in the catalog (the record's are 0), or `None` for any other file.
fn foreign_flac(path: &Path) -> Result<Option<Hydrated>, TierError> {
    use std::io::Read;
    let mut magic = [0u8; 4];
    let mut file = std::fs::File::open(path)?;
    if file.read_exact(&mut magic).is_err() || !crate::v1audio::is_flac(&magic) {
        return Ok(None);
    }
    // v1's files are minutes of audio, tens of MB; a path past this was registered by mistake.
    const MAX_FOREIGN_BYTES: u64 = 1 << 30;
    let size = file.metadata()?.len();
    if size > MAX_FOREIGN_BYTES {
        return Err(TierError::Codec(format!("{}: {size} bytes is too big for a v1 FLAC file", path.display())));
    }
    let bytes = std::fs::read(path)?;
    let len = u32::try_from(bytes.len()).map_err(|_| TierError::Codec(format!("{}: too big", path.display())))?;
    let record =
        Record { start_ns: 0, end_ns: 0, seq: 0, offset: 0, len, cut: false, id: continuum_core::Uuid([0; 16]) };
    Ok(Some(Hydrated { index: 0, record, bytes, pcm: None }))
}

/// An MP4 is registered in place, never written into a chunk file: its GOPs are read through [`crate::v1video`].
fn mp4_in_a_chunk() -> TierError {
    TierError::Codec("an MP4 frame in a chunk file: read v1 recordings with v1video::gop_units".to_owned())
}

/// A decoded picture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Picture {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Interleaved 8 bit RGB, row-major, no padding.
    pub rgb: Vec<u8>,
}

/// Decodes a JPEG frame to RGB on the CPU (the fallback for an MJPEG camera when no hardware decoder is built in).
///
/// # Errors
///
/// [`TierError::Codec`] for bytes that aren't a JPEG.
pub fn decode_jpeg(bytes: &[u8]) -> Result<Picture, TierError> {
    use zune_jpeg::JpegDecoder;
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;

    // Strict: a truncated or malformed frame is an error, never a picture with a grey band nobody noticed.
    let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB).set_strict_mode(true);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(bytes), options);
    let rgb = decoder.decode().map_err(|e| TierError::Codec(format!("jpeg: {e}")))?;
    let info = decoder.info().ok_or_else(|| TierError::Codec("jpeg: no image info".to_owned()))?;
    Ok(Picture { width: u32::from(info.width), height: u32::from(info.height), rgb })
}

/// Converts RGB to NV12 (full-range BT.601, 4:2:0 with interleaved UV), the format the doc names for decoded
/// frames. Dimensions should be even: an odd width gives a chroma plane whose stride is one more than half the
/// luma's, which consumers that assume equal strides misread (the chroma average repeats the last row or column).
#[must_use]
pub fn rgb_to_nv12(picture: &Picture) -> Vec<u8> {
    let (w, h) = (picture.width as usize, picture.height as usize);
    let mut out = vec![0u8; w * h + 2 * w.div_ceil(2) * h.div_ceil(2)];
    let px = |x: usize, y: usize| -> (f32, f32, f32) {
        let i = (y.min(h.saturating_sub(1)) * w + x.min(w.saturating_sub(1))) * 3;
        (f32::from(picture.rgb[i]), f32::from(picture.rgb[i + 1]), f32::from(picture.rgb[i + 2]))
    };
    for y in 0..h {
        for x in 0..w {
            let (r, g, b) = px(x, y);
            let luma = 0.299 * r + 0.587 * g + 0.114 * b;
            out[y * w + x] = luma.round().clamp(0.0, 255.0) as u8;
        }
    }
    let uv_base = w * h;
    let uv_w = w.div_ceil(2);
    for cy in 0..h.div_ceil(2) {
        for cx in 0..uv_w {
            let mut sum = (0.0f32, 0.0f32, 0.0f32);
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let (r, g, b) = px(cx * 2 + dx, cy * 2 + dy);
                sum = (sum.0 + r, sum.1 + g, sum.2 + b);
            }
            let (r, g, b) = (sum.0 / 4.0, sum.1 / 4.0, sum.2 / 4.0);
            let u = 128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b;
            let v = 128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b;
            let i = uv_base + (cy * uv_w + cx) * 2;
            out[i] = u.round().clamp(0.0, 255.0) as u8;
            out[i + 1] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Converts NV12 (BT.601, 4:2:0 with interleaved UV, chroma at the same stride as luma) to an RGB picture: `limited`
/// for a camera's video range (luma 16 to 235, chroma 16 to 240, as raw cameras send and `crate::raw` keeps), else
/// full range (pictures from JPEG). Returns None for a buffer too short for the size.
#[must_use]
pub fn nv12_to_rgb(nv12: &[u8], width: u32, height: u32, limited: bool) -> Option<Picture> {
    let (w, h) = (width as usize, height as usize);
    let uv_w = w.div_ceil(2);
    if nv12.len() < w * h + 2 * uv_w * h.div_ceil(2) {
        return None;
    }
    let (y_scale, y_off, c_scale) = if limited { (255.0 / 219.0, 16.0, 255.0 / 224.0) } else { (1.0, 0.0, 1.0) };
    let mut rgb = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let luma = (f32::from(nv12[y * w + x]) - y_off) * y_scale;
            let i = w * h + ((y / 2) * uv_w + x / 2) * 2;
            let (u, v) = ((f32::from(nv12[i]) - 128.0) * c_scale, (f32::from(nv12[i + 1]) - 128.0) * c_scale);
            let px = [luma + 1.402 * v, luma - 0.344_136 * u - 0.714_136 * v, luma + 1.772 * u];
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            rgb.extend(px.map(|c| c.round().clamp(0.0, 255.0) as u8));
        }
    }
    Some(Picture { width, height, rgb })
}

#[cfg(test)]
mod tests {
    #[test]
    fn nv12_reads_back_as_rgb_in_either_range() {
        // A 2x2 frame: one luma value, neutral chroma.
        let frame = |y: u8| vec![y, y, y, y, 128, 128];
        let grey = |p: Picture| p.rgb.iter().copied().max().unwrap();
        assert_eq!(grey(nv12_to_rgb(&frame(16), 2, 2, true).unwrap()), 0, "limited range: 16 is black");
        assert_eq!(grey(nv12_to_rgb(&frame(235), 2, 2, true).unwrap()), 255, "and 235 white");
        assert_eq!(grey(nv12_to_rgb(&frame(200), 2, 2, false).unwrap()), 200, "full range: as it is");
        assert_eq!(nv12_to_rgb(&[0; 5], 2, 2, true), None, "too short");
    }

    use super::*;

    fn id(n: u8) -> continuum_core::Uuid {
        continuum_core::Uuid([n; 16])
    }

    /// A chunk of two payloads, 1 and 2: the path and their records.
    fn two_payloads(dir: &Path) -> (std::path::PathBuf, Vec<Record>) {
        let path = dir.join("c.cnk");
        let mut w = crate::chunk::ChunkWriter::create(&path, continuum_core::StreamId(1), ChunkFormat::Passthrough, 0)
            .expect("create");
        w.append_payload(id(1), 100, 200, 0, b"first").expect("append");
        w.append_payload(id(2), 200, 300, 0, b"second").expect("append");
        let records = w.close().expect("close");
        (path, records)
    }

    #[test]
    fn a_chunk_reads_as_its_live_records_in_the_range() {
        let dir = tempfile::tempdir().expect("dir");
        let (path, _) = two_payloads(dir.path());
        let got = |s, e| {
            read_chunk(&path, s, e)
                .expect("read")
                .expect("there")
                .into_iter()
                .map(|h| (h.record.id, h.bytes))
                .collect::<Vec<_>>()
        };
        assert_eq!(got(0, 1000), vec![(id(1), b"first".to_vec()), (id(2), b"second".to_vec())]);
        assert_eq!(got(250, 1000), vec![(id(2), b"second".to_vec())], "only what overlaps the range");
        crate::chunk::rewrite_without(&path, &[0]).expect("purge");
        assert_eq!(got(0, 1000), vec![(id(2), b"second".to_vec())], "a record a purge cut is left out");
    }

    #[test]
    fn a_chunk_that_is_gone_reads_as_none_and_a_corrupt_one_is_an_error() {
        let dir = tempfile::tempdir().expect("dir");
        let (path, records) = two_payloads(dir.path());
        let mut bytes = std::fs::read(&path).expect("read");
        let at = usize::try_from(records[0].offset).expect("fits");
        bytes[at] ^= 0xff;
        std::fs::write(&path, bytes).expect("write");
        assert!(read_chunk(&path, 0, 1000).is_err(), "corruption is reported, never read as gone");
        std::fs::remove_file(&path).expect("unlink");
        assert!(read_chunk(&path, 0, 1000).expect("read").is_none());
    }

    #[test]
    fn nv12_of_a_flat_grey_picture_is_flat() {
        let picture = Picture { width: 4, height: 3, rgb: vec![100; 4 * 3 * 3] };
        let nv12 = rgb_to_nv12(&picture);
        assert_eq!(nv12.len(), 12 + 2 * 2 * 2);
        assert!(nv12[..12].iter().all(|&y| y == 100));
        assert!(nv12[12..].iter().all(|&c| c == 128));
    }

    #[test]
    fn nv12_of_pure_red_has_high_v_and_low_u() {
        let mut rgb = Vec::new();
        for _ in 0..4 {
            rgb.extend_from_slice(&[255, 0, 0]);
        }
        let nv12 = rgb_to_nv12(&Picture { width: 2, height: 2, rgb });
        assert_eq!(nv12[..4], [76, 76, 76, 76]);
        assert!(nv12[4] < 100 && nv12[5] > 200, "{:?}", &nv12[4..]);
    }

    #[test]
    fn not_a_jpeg_is_refused() {
        assert!(decode_jpeg(b"definitely not a jpeg").is_err());
    }
}
