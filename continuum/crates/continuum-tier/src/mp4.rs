//! Reading the sample table of an ordinary (unfragmented) MP4's video track, in pure Rust, for registering
//! recordings in place ([`crate::inplace_video`]).
//!
//! Only the `moov` box is read (the `mdat` is skipped by seeking): the track's timescale, its sample entry (codec,
//! size, and the `avcC` or `hvcC` that holds the parameter sets), and from its sample table each sample's byte
//! offset (`stsc` with `stco` or `co64`), size (`stsz`), decode time (`stts`), composition offset (`ctts`, for
//! B-frames), and whether it's a sync sample (`stss`). The first edit (`elst`) says where presentation starts, which
//! FFmpeg's muxer sets to skip the B-frame delay. Every count and size read from the file is bounded before it's
//! used, so a damaged or hostile file is an error, not a huge allocation.

use std::io::{Read, Seek, SeekFrom};

use crate::TierError;

/// The most a `moov` box may be: a 5 minute 30 fps recording's is about 110 KB.
const MAX_MOOV: u64 = 64 << 20;
/// The most samples a track may have: about 4 days at 30 fps.
const MAX_SAMPLES: u64 = 10_000_000;

/// One sample (an access unit) of the track, in decode order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sample {
    /// Its first byte in the file.
    pub offset: u64,
    /// Its length.
    pub size: u32,
    /// Its decode time, in the track's timescale.
    pub dts: u64,
    /// Its presentation time, in the track's timescale, the first edit applied (so the first frame shown is at 0).
    pub pts: i64,
    /// Its duration, in the track's timescale.
    pub duration: u32,
    /// A sync sample (keyframe).
    pub key: bool,
}

/// A file's video track.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mp4Video {
    /// Units per second of the sample times.
    pub timescale: u32,
    /// The sample entry's four-character code: `avc1`, `avc3`, `hvc1`, or `hev1`.
    pub codec: [u8; 4],
    /// Width in pixels.
    pub width: u16,
    /// Height in pixels.
    pub height: u16,
    /// The decoder configuration record (`avcC` or `hvcC`'s payload): the parameter sets and the NAL length size.
    pub config: Vec<u8>,
    /// The samples, in decode order.
    pub samples: Vec<Sample>,
    /// Where presentation starts in the media's decode timeline (the first edit's media time): a sample's decode
    /// time on the presentation clock is `dts - media_start`.
    pub media_start: i64,
}

/// A group of pictures: a keyframe and the samples after it up to the next, in decode order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gop {
    /// Its first sample's index (the keyframe).
    pub first: usize,
    /// How many samples it has.
    pub count: usize,
    /// The earliest presentation time among them, in the track's timescale.
    pub pts: i64,
    /// The latest presentation end among them, in the track's timescale.
    pub pts_end: i64,
    /// The first byte of its samples in the file.
    pub offset: u64,
    /// The bytes from `offset` to the end of its last sample.
    pub length: u64,
}

impl Mp4Video {
    /// The groups of pictures, in decode order. Samples before the first keyframe (none, in a file a muxer wrote) are
    /// left out, since nothing can decode them.
    #[must_use]
    pub fn gops(&self) -> Vec<Gop> {
        let mut out: Vec<Gop> = Vec::new();
        for (i, s) in self.samples.iter().enumerate() {
            let end = s.pts.saturating_add(i64::from(s.duration));
            if s.key {
                out.push(Gop { first: i, count: 0, pts: s.pts, pts_end: end, offset: s.offset, length: 0 });
            }
            let Some(g) = out.last_mut() else { continue };
            g.count += 1;
            g.pts = g.pts.min(s.pts);
            g.pts_end = g.pts_end.max(end);
            let lo = g.offset.min(s.offset);
            let hi = (g.offset + g.length).max(s.offset + u64::from(s.size));
            g.offset = lo;
            g.length = hi - lo;
        }
        out
    }

    /// The track's length in the timescale: the latest presentation end.
    #[must_use]
    pub fn duration(&self) -> i64 {
        self.samples.iter().map(|s| s.pts.saturating_add(i64::from(s.duration))).max().unwrap_or(0)
    }
}

fn bad(why: impl Into<String>) -> TierError {
    TierError::Codec(format!("mp4: {}", why.into()))
}

/// A box's type and its payload's place: `start` (just past the header) and `len`.
struct BoxHead {
    kind: [u8; 4],
    start: u64,
    len: u64,
}

/// Reads the box header at `at` in a stream `total` bytes long.
fn read_head(r: &mut (impl Read + Seek), at: u64, total: u64) -> Result<BoxHead, TierError> {
    r.seek(SeekFrom::Start(at))?;
    let mut h = [0u8; 8];
    r.read_exact(&mut h)?;
    let size32 = u32::from_be_bytes([h[0], h[1], h[2], h[3]]);
    let kind = [h[4], h[5], h[6], h[7]];
    let (header, size) = match size32 {
        0 => (8, total - at),
        1 => {
            let mut l = [0u8; 8];
            r.read_exact(&mut l)?;
            (16, u64::from_be_bytes(l))
        }
        n => (8, u64::from(n)),
    };
    if size < header || at.checked_add(size).is_none_or(|end| end > total) {
        return Err(bad(format!("box {} at {at} runs past the file", String::from_utf8_lossy(&kind))));
    }
    Ok(BoxHead { kind, start: at + header, len: size - header })
}

/// The boxes inside `bytes` (a container's payload), as (type, payload).
/// A container's child boxes: each one's type and payload.
type Children<'a> = Vec<([u8; 4], &'a [u8])>;

fn children(bytes: &[u8]) -> Result<Children<'_>, TierError> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 8 <= bytes.len() {
        let size32 = u32::from_be_bytes(bytes[i..i + 4].try_into().expect("4 bytes")) as usize;
        let kind: [u8; 4] = bytes[i + 4..i + 8].try_into().expect("4 bytes");
        let (header, size) = match size32 {
            0 => (8, bytes.len() - i),
            1 if i + 16 <= bytes.len() => {
                let s = u64::from_be_bytes(bytes[i + 8..i + 16].try_into().expect("8 bytes"));
                (16, usize::try_from(s).map_err(|_| bad("a box too big"))?)
            }
            1 => return Err(bad("a truncated large box")),
            n => (8, n),
        };
        if size < header || i.checked_add(size).is_none_or(|end| end > bytes.len()) {
            return Err(bad(format!("box {} overruns its parent", String::from_utf8_lossy(&kind))));
        }
        out.push((kind, &bytes[i + header..i + size]));
        i += size;
    }
    Ok(out)
}

fn child<'a>(bytes: &'a [u8], kind: &[u8; 4]) -> Result<Option<&'a [u8]>, TierError> {
    Ok(children(bytes)?.into_iter().find(|(k, _)| k == kind).map(|(_, b)| b))
}

fn need<'a>(bytes: &'a [u8], kind: &[u8; 4]) -> Result<&'a [u8], TierError> {
    child(bytes, kind)?.ok_or_else(|| bad(format!("no {} box", String::from_utf8_lossy(kind))))
}

/// A cursor over a box's payload that refuses to read past it.
struct Cur<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Cur<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, i: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], TierError> {
        let end = self.i.checked_add(n).filter(|&e| e <= self.b.len()).ok_or_else(|| bad("a truncated table"))?;
        let s = &self.b[self.i..end];
        self.i = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, TierError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, TierError> {
        Ok(u16::from_be_bytes(self.take(2)?.try_into().expect("2 bytes")))
    }
    fn u32(&mut self) -> Result<u32, TierError> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into().expect("4 bytes")))
    }
    fn u64(&mut self) -> Result<u64, TierError> {
        Ok(u64::from_be_bytes(self.take(8)?.try_into().expect("8 bytes")))
    }
    /// A full box's version, after which its 24 bits of flags are skipped.
    fn version(&mut self) -> Result<u8, TierError> {
        let v = self.u8()?;
        self.take(3)?;
        Ok(v)
    }
    /// An entry count, refused if the remaining bytes can't hold that many entries of `entry` bytes.
    fn count(&mut self, entry: usize) -> Result<usize, TierError> {
        let n = self.u32()? as usize;
        if n.checked_mul(entry).is_none_or(|need| need > self.b.len() - self.i) {
            return Err(bad(format!("{n} entries don't fit their table")));
        }
        Ok(n)
    }
}

/// Reads the video track of the MP4 in `r`.
///
/// # Errors
///
/// [`TierError::Codec`] for a file with no `moov` (one still being written), no video track, a damaged or
/// inconsistent sample table, or an unsupported codec; an I/O error as it comes.
pub fn read_video(r: &mut (impl Read + Seek)) -> Result<Mp4Video, TierError> {
    let total = r.seek(SeekFrom::End(0))?;
    let mut at = 0;
    let moov = loop {
        if at + 8 > total {
            return Err(bad("no moov box (the file was never finished)"));
        }
        let h = read_head(r, at, total)?;
        if &h.kind == b"moov" {
            if h.len > MAX_MOOV {
                return Err(bad(format!("a moov of {} bytes", h.len)));
            }
            let mut buf = vec![0u8; usize::try_from(h.len).map_err(|_| bad("moov too big"))?];
            r.seek(SeekFrom::Start(h.start))?;
            r.read_exact(&mut buf)?;
            break buf;
        }
        at = h.start + h.len;
    };
    for (kind, trak) in children(&moov)? {
        if &kind != b"trak" {
            continue;
        }
        let mdia = need(trak, b"mdia")?;
        let hdlr = need(mdia, b"hdlr")?;
        if hdlr.get(8..12) != Some(b"vide") {
            continue;
        }
        let movie_timescale = match child(&moov, b"mvhd")? {
            Some(mvhd) => {
                let mut c = Cur::new(mvhd);
                if c.version()? == 1 {
                    c.take(16)?
                } else {
                    c.take(8)?
                };
                c.u32()?
            }
            None => 0,
        };
        return read_track(trak, mdia, movie_timescale, total);
    }
    Err(bad("no video track"))
}

fn read_track(trak: &[u8], mdia: &[u8], movie_timescale: u32, total: u64) -> Result<Mp4Video, TierError> {
    let mut mdhd = Cur::new(need(mdia, b"mdhd")?);
    let timescale = if mdhd.version()? == 1 {
        mdhd.take(16)?;
        mdhd.u32()?
    } else {
        mdhd.take(8)?;
        mdhd.u32()?
    };
    if timescale == 0 {
        return Err(bad("a timescale of 0"));
    }
    // The first edit's media time: where presentation starts (FFmpeg skips the B-frame delay with it). Empty edits
    // before it delay presentation by their duration, in the movie's timescale.
    let mut media_start = 0i64;
    let mut delay = 0u64;
    if let Some(edts) = child(trak, b"edts")?
        && let Some(elst) = child(edts, b"elst")?
    {
        let mut c = Cur::new(elst);
        let v = c.version()?;
        let n = c.count(if v == 1 { 20 } else { 12 })?;
        for _ in 0..n {
            let (duration, media_time) = if v == 1 {
                (c.u64()?, i64::from_be_bytes(c.take(8)?.try_into().expect("8 bytes")))
            } else {
                (u64::from(c.u32()?), i64::from(i32::from_be_bytes(c.take(4)?.try_into().expect("4 bytes"))))
            };
            c.take(4)?;
            // An empty edit (-1) delays the start; the first real one says where the media starts.
            if media_time >= 0 {
                media_start = media_time;
                break;
            }
            delay = delay.saturating_add(duration);
        }
        if delay > 0 && movie_timescale > 0 {
            let in_media = u128::from(delay) * u128::from(timescale) / u128::from(movie_timescale);
            let in_media = i64::try_from(in_media).map_err(|_| bad("an empty edit too long"))?;
            media_start = media_start.checked_sub(in_media).ok_or_else(|| bad("an empty edit too long"))?;
        }
    }
    let stbl = need(need(mdia, b"minf")?, b"stbl")?;
    // The sample entry: the first in stsd.
    let mut stsd = Cur::new(need(stbl, b"stsd")?);
    stsd.version()?;
    if stsd.count(8)? == 0 {
        return Err(bad("no sample entry"));
    }
    let entries = children(&stsd.b[stsd.i..])?;
    let (codec, entry) = *entries.first().ok_or_else(|| bad("no sample entry"))?;
    let config_kind: &[u8; 4] = match &codec {
        b"avc1" | b"avc3" => b"avcC",
        b"hvc1" | b"hev1" => b"hvcC",
        other => return Err(bad(format!("codec {} isn't H.264 or HEVC", String::from_utf8_lossy(other)))),
    };
    // A visual sample entry: 6 reserved, 2 data reference index, 16 predefined and reserved, then width and height,
    // and 50 more bytes before its child boxes.
    let mut e = Cur::new(entry);
    e.take(24)?;
    let (width, height) = (e.u16()?, e.u16()?);
    e.take(50)?;
    let config = need(&entry[e.i..], config_kind)?.to_vec();
    let samples = sample_table(stbl, media_start, total)?;
    Ok(Mp4Video { timescale, codec, width, height, config, samples, media_start })
}

/// The samples of `stbl`, in a file `total` bytes long: every sample must lie inside it, which also bounds how many
/// a small table can claim.
fn sample_table(stbl: &[u8], media_start: i64, total: u64) -> Result<Vec<Sample>, TierError> {
    // Sizes.
    let mut stsz = Cur::new(need(stbl, b"stsz")?);
    stsz.version()?;
    let uniform = stsz.u32()?;
    let n = stsz.u32()? as usize;
    if n as u64 > MAX_SAMPLES {
        return Err(bad(format!("{n} samples")));
    }
    if uniform != 0 && (n as u64).saturating_mul(u64::from(uniform)) > total {
        return Err(bad(format!("{n} samples of {uniform} bytes don't fit a file of {total}")));
    }
    let sizes: Vec<u32> = if uniform == 0 {
        if n.checked_mul(4).is_none_or(|need| need > stsz.b.len() - stsz.i) {
            return Err(bad("stsz is shorter than its sample count"));
        }
        (0..n).map(|_| stsz.u32()).collect::<Result<_, _>>()?
    } else {
        vec![uniform; n]
    };
    // Decode times and durations.
    let mut stts = Cur::new(need(stbl, b"stts")?);
    stts.version()?;
    let mut durations = Vec::with_capacity(n);
    for _ in 0..stts.count(8)? {
        let (count, delta) = (stts.u32()? as usize, stts.u32()?);
        if durations.len() + count > n {
            return Err(bad("stts names more samples than stsz"));
        }
        durations.extend(std::iter::repeat_n(delta, count));
    }
    if durations.len() != n {
        return Err(bad("stts and stsz disagree on the sample count"));
    }
    // Composition offsets (B-frames); none means presentation is decode order.
    let mut offsets = vec![0i64; n];
    if let Some(ctts) = child(stbl, b"ctts")? {
        let mut c = Cur::new(ctts);
        let v = c.version()?;
        let mut i = 0usize;
        for _ in 0..c.count(8)? {
            let count = c.u32()? as usize;
            let raw = c.u32()?;
            // Version 1 offsets are signed; FFmpeg writes version 0 with values that only make sense signed too.
            let off = if v == 1 || raw > i32::MAX as u32 { i64::from(raw as i32) } else { i64::from(raw) };
            let end = i.checked_add(count).filter(|&e| e <= n).ok_or_else(|| bad("ctts names more samples"))?;
            offsets[i..end].fill(off);
            i = end;
        }
    }
    // Sync samples; none means every sample is one.
    let keys: Option<Vec<bool>> = match child(stbl, b"stss")? {
        None => None,
        Some(stss) => {
            let mut c = Cur::new(stss);
            c.version()?;
            let mut k = vec![false; n];
            for _ in 0..c.count(4)? {
                let s = c.u32()? as usize;
                *k.get_mut(s.checked_sub(1).ok_or_else(|| bad("stss sample 0"))?)
                    .ok_or_else(|| bad("stss past the end"))? = true;
            }
            Some(k)
        }
    };
    // Chunk offsets.
    let chunk_offsets: Vec<u64> = if let Some(stco) = child(stbl, b"stco")? {
        let mut c = Cur::new(stco);
        c.version()?;
        (0..c.count(4)?).map(|_| c.u32().map(u64::from)).collect::<Result<_, _>>()?
    } else {
        let mut c = Cur::new(need(stbl, b"co64")?);
        c.version()?;
        (0..c.count(8)?).map(|_| c.u64()).collect::<Result<_, _>>()?
    };
    // Samples per chunk, as runs: (first chunk, 1-based; samples per chunk).
    let mut stsc = Cur::new(need(stbl, b"stsc")?);
    stsc.version()?;
    let runs: Vec<(usize, usize)> = (0..stsc.count(12)?)
        .map(|_| Ok((stsc.u32()? as usize, stsc.u32()? as usize, stsc.u32()?)))
        .map(|r: Result<(usize, usize, u32), TierError>| r.map(|(first, per, _)| (first, per)))
        .collect::<Result<_, _>>()?;
    let mut offsets_of = Vec::with_capacity(n);
    for (k, &(first, per)) in runs.iter().enumerate() {
        let next = runs.get(k + 1).map(|&(next, _)| next);
        if first == 0 || next.is_some_and(|next| next <= first) || first > chunk_offsets.len() {
            return Err(bad("stsc runs that don't rise within the chunk table"));
        }
        let last = next.map_or(chunk_offsets.len(), |next| (next - 1).min(chunk_offsets.len()));
        for chunk in first..=last {
            let mut at = chunk_offsets[chunk - 1];
            for _ in 0..per {
                let Some(&size) = sizes.get(offsets_of.len()) else { break };
                let end = at.checked_add(u64::from(size)).filter(|&e| e <= total);
                let end = end.ok_or_else(|| bad(format!("a sample at {at} runs past the file's {total} bytes")))?;
                offsets_of.push(at);
                at = end;
            }
        }
    }
    if offsets_of.len() != n {
        return Err(bad(format!("the chunk tables place {} of {n} samples", offsets_of.len())));
    }
    let mut out = Vec::with_capacity(n);
    let mut dts = 0u64;
    for i in 0..n {
        let pts = i64::try_from(dts)
            .ok()
            .and_then(|d| d.checked_add(offsets[i]))
            .and_then(|t| t.checked_sub(media_start))
            .ok_or_else(|| bad("a presentation time overflows"))?;
        out.push(Sample {
            offset: offsets_of[i],
            size: sizes[i],
            dts,
            pts,
            duration: durations[i],
            key: keys.as_ref().is_none_or(|k| k[i]),
        });
        dts = dts.checked_add(u64::from(durations[i])).ok_or_else(|| bad("a decode time overflows"))?;
    }
    Ok(out)
}

/// The AVCC samples of an H.264 or HEVC track as Annex B: each length-prefixed NAL unit given a start code, with the
/// parameter sets from `config` put in front of a keyframe so it decodes on its own.
///
/// # Errors
///
/// [`TierError::Codec`] for a sample whose NAL lengths overrun it, or a configuration record that can't be read.
pub fn to_annex_b(video: &Mp4Video, sample: &[u8], key: bool) -> Result<Vec<u8>, TierError> {
    let (length_size, sets) = parameter_sets(video)?;
    let mut out = Vec::with_capacity(sample.len() + 64);
    if key {
        for s in &sets {
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(s);
        }
    }
    let mut i = 0usize;
    while i < sample.len() {
        let head = sample.get(i..i + length_size).ok_or_else(|| bad("a NAL length past the sample"))?;
        let len = head.iter().fold(0usize, |a, &b| (a << 8) | usize::from(b));
        i += length_size;
        let nal = sample.get(i..i + len).ok_or_else(|| bad("a NAL unit past the sample"))?;
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(nal);
        i += len;
    }
    Ok(out)
}

/// The NAL length size and the parameter sets (VPS, SPS, and PPS) from an `avcC` or `hvcC` record.
fn parameter_sets(video: &Mp4Video) -> Result<(usize, Vec<Vec<u8>>), TierError> {
    let mut c = Cur::new(&video.config);
    let mut sets = Vec::new();
    match &video.codec {
        b"avc1" | b"avc3" => {
            c.take(4)?;
            let length_size = usize::from(c.u8()? & 0x03) + 1;
            let sps = usize::from(c.u8()? & 0x1f);
            for _ in 0..sps {
                let n = usize::from(c.u16()?);
                sets.push(c.take(n)?.to_vec());
            }
            for _ in 0..c.u8()? {
                let n = usize::from(c.u16()?);
                sets.push(c.take(n)?.to_vec());
            }
            Ok((length_size, sets))
        }
        _ => {
            c.take(21)?;
            let length_size = usize::from(c.u8()? & 0x03) + 1;
            for _ in 0..c.u8()? {
                c.take(1)?;
                for _ in 0..c.u16()? {
                    let n = usize::from(c.u16()?);
                    sets.push(c.take(n)?.to_vec());
                }
            }
            Ok((length_size, sets))
        }
    }
}
