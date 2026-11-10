//! The opt-in AV1 archive pass (#1479; docs/STORAGE.md, "AV1 archive"), behind the `ffmpeg` feature.
//!
//! [`encode_chunk`] turns one closed [`ChunkFormat::H264Au`] chunk into AV1: it decodes the chunk's H.264 in
//! software and encodes each picture with SVT-AV1 (preset 8, CRF 52 by default, the measured recommendation), one
//! temporal unit per record. Pictures stream through, so at most a few are held at a time. [`Encoded::swap`] then
//! writes the [`ChunkFormat::Av1Au`] chunk and renames it over the original, refusing if the file changed since it
//! was read (a purge rewrote it), so the catalog's frame numbers, payload ids, and times all stay as they were; the
//! caller tells the catalog the new offsets with `RewriteChunk`.
//!
//! Each chunk is its own GOP: a chunk decodes on its own, so its first unit is its only planned keyframe. With the
//! tier worker's chunks of about 3.5 seconds, that costs about 20% more bytes than the 10-second GOPs measured
//! (docs/STORAGE.md), for no change to the catalog or its readers' chunk rules.
//!
//! A chunk stays H.264 ([`Encoded::Kept`]) when converting it would change what readers see: tombstones (a
//! purge's cut leaves the rest of its H.264 GOP undecodable), frames out of time order (B-frames, which AV1 units
//! can't keep in the record order), a first frame that isn't a keyframe, or any frame that doesn't decode.

use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use ff::util::color::Range;
use ffmpeg_next as ff;

use crate::TierError;
use crate::chunk::{ChunkFormat, ChunkReader, Rewritten, rewrite_as};
use crate::encode::{Av1Options, Colors, Encoder, Settings};
use crate::gop::{self, Unit};

/// The archive encoder's controls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Av1Settings {
    /// SVT-AV1 preset, 0 (slowest, smallest) to 13.
    pub preset: u8,
    /// Constant rate factor, 0 to 63: higher is smaller and softer.
    pub crf: u8,
    /// SVT-AV1's level of parallelism: about how many cores one chunk's encode keeps busy.
    pub parallelism: u32,
}

impl Default for Av1Settings {
    /// Preset 8 and CRF 52, the measured recommendation (docs/STORAGE.md), on up to 4 cores.
    fn default() -> Self {
        Self { preset: 8, crf: 52, parallelism: 4 }
    }
}

/// What [`encode_chunk`] made of a chunk.
#[derive(Debug)]
pub enum Encoded {
    /// The chunk's AV1 records, ready for [`Encoded::swap`].
    Ready(Ready),
    /// The chunk stays as it is, and why.
    Kept(String),
}

/// A chunk's AV1 records, and the file they were made from.
#[derive(Debug)]
pub struct Ready {
    /// One entry per record, in index order: `None` for a tombstone (there are none: tombstoned chunks are kept).
    bytes: Vec<Option<Vec<u8>>>,
    source: Stamp,
    /// The source file's length.
    pub before: u64,
    /// Frames converted.
    pub frames: usize,
}

/// What identifies the file as read: a rewrite renames a new file in, so its inode, length, or time changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stamp {
    len: u64,
    modified: Option<std::time::SystemTime>,
    #[cfg(unix)]
    ino: u64,
}

impl Stamp {
    fn of(path: &Path) -> std::io::Result<Self> {
        let meta = fs::metadata(path)?;
        Ok(Self {
            len: meta.len(),
            modified: meta.modified().ok(),
            #[cfg(unix)]
            ino: std::os::unix::fs::MetadataExt::ino(&meta),
        })
    }
}

impl Ready {
    /// Writes the AV1 chunk and renames it over `path`, unless `path` changed since [`encode_chunk`] read it (then
    /// `None`, and the file is left alone).
    ///
    /// # Errors
    ///
    /// A read or write error.
    pub fn swap(&self, path: &Path) -> Result<Option<Rewritten>, TierError> {
        match Stamp::of(path) {
            Ok(now) if now == self.source => {}
            Ok(_) => return Ok(None),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        }
        Ok(Some(rewrite_as(path, ChunkFormat::Av1Au, &self.bytes)?))
    }
}

fn kept(why: impl Into<String>) -> Result<Encoded, TierError> {
    Ok(Encoded::Kept(why.into()))
}

/// Re-encodes a closed [`ChunkFormat::H264Au`] chunk's pictures as AV1 (see the module docs). `stop` set partway
/// ends it early as [`Encoded::Kept`].
///
/// # Errors
///
/// A read error, or an encoder that won't open or fails (a codec problem with this machine, not this chunk). A
/// chunk whose own frames don't decode is [`Encoded::Kept`].
pub fn encode_chunk(path: &Path, settings: Av1Settings, stop: &AtomicBool) -> Result<Encoded, TierError> {
    let source = Stamp::of(path)?;
    let mut reader = ChunkReader::open(path)?;
    if reader.format() != ChunkFormat::H264Au {
        return kept(format!("{:?}, not H.264", reader.format()));
    }
    let records = reader.records().to_vec();
    if records.is_empty() {
        return kept("no frames");
    }
    if records.iter().any(|r| r.cut) {
        return kept("purged frames");
    }
    let mut units = Vec::with_capacity(records.len());
    for i in 0..records.len() {
        let mut got = gop::decode(&reader.read(i)?)?;
        if got.len() != 1 {
            return kept("a record with more than one access unit");
        }
        units.push(got.remove(0));
    }
    drop(reader);
    if units.windows(2).any(|w| w[1].start_ns < w[0].start_ns) {
        return kept("frames out of time order");
    }
    if !units[0].key {
        return kept("no keyframe first");
    }
    let mut pass = Pass { settings, units: &units, rate: frame_rate(&units), encoder: None, out: Vec::new(), next: 0 };
    let mut decoder = {
        let _open = crate::encode::codec_open().map_err(|e| TierError::Codec(format!("h264 decoder init: {e}")))?;
        let found =
            ff::decoder::find(ff::codec::Id::H264).ok_or_else(|| TierError::Codec("no H.264 decoder".into()))?;
        ff::codec::context::Context::new_with_codec(found)
            .decoder()
            .video()
            .map_err(|e| TierError::Codec(format!("h264 decoder open: {e}")))?
    };
    for (i, unit) in units.iter().enumerate() {
        if stop.load(Ordering::Relaxed) {
            return kept("stopped");
        }
        let mut packet = ff::Packet::copy(&unit.bytes);
        packet.set_pts(Some(i as i64));
        if let Err(e) = decoder.send_packet(&packet) {
            return kept(format!("frame {i} doesn't decode: {e}"));
        }
        if let Some(why) = pass.take(&mut decoder)? {
            return kept(why);
        }
    }
    if decoder.send_eof().is_err() {
        return kept("the decoder didn't finish");
    }
    if let Some(why) = pass.take(&mut decoder)? {
        return kept(why);
    }
    if pass.next != units.len() {
        return kept(format!("{} of {} frames decoded", pass.next, units.len()));
    }
    let Some(encoder) = pass.encoder.take() else { return kept("no frames decoded") };
    for au in encoder.finish()? {
        pass.out.push(Some(record(&au)?));
    }
    if pass.out.len() != units.len() {
        return Err(TierError::Codec(format!("av1 returned {} units for {} frames", pass.out.len(), units.len())));
    }
    Ok(Encoded::Ready(Ready { bytes: pass.out, source, before: source.len, frames: units.len() }))
}

/// One chunk's decode-then-encode, picture by picture.
struct Pass<'a> {
    settings: Av1Settings,
    units: &'a [Unit],
    rate: u32,
    encoder: Option<Encoder>,
    out: Vec<Option<Vec<u8>>>,
    /// The record the next decoded picture must be.
    next: usize,
}

impl Pass<'_> {
    /// Encodes every picture the decoder has ready. `Some(why)` when this chunk can't be converted.
    fn take(&mut self, decoder: &mut ff::decoder::Video) -> Result<Option<String>, TierError> {
        let mut picture = ff::frame::Video::empty();
        loop {
            match decoder.receive_frame(&mut picture) {
                Ok(()) => {}
                Err(ff::Error::Other { errno: ff::util::error::EAGAIN } | ff::Error::Eof) => return Ok(None),
                Err(e) => return Ok(Some(format!("a frame doesn't decode: {e}"))),
            }
            if picture.pts() != i64::try_from(self.next).ok() {
                return Ok(Some(format!("frame {} didn't decode", self.next)));
            }
            crate::video::normalize_decoded_range(&mut picture);
            if self.encoder.is_none() {
                let colors = Colors {
                    range: if picture.color_range() == Range::JPEG { Range::JPEG } else { Range::MPEG },
                    space: picture.color_space(),
                    primaries: picture.color_primaries(),
                    transfer: picture.color_transfer_characteristic(),
                };
                let settings = Settings {
                    gop_frames: u32::try_from(self.units.len()).unwrap_or(u32::MAX),
                    frame_rate: self.rate,
                    bit_rate: 0,
                    full_range: colors.range == Range::JPEG,
                    low_delay: false,
                    threads: 0,
                };
                let options = Av1Options {
                    preset: self.settings.preset,
                    crf: self.settings.crf,
                    parallelism: self.settings.parallelism,
                };
                self.encoder = Some(Encoder::open_av1(picture.width(), picture.height(), settings, options, colors)?);
            }
            let unit = &self.units[self.next];
            let nv12 = crate::video::to_nv12(&picture)?;
            let encoder = self.encoder.as_mut().expect("opened above");
            for au in encoder.encode_keyframe(&nv12, unit.start_ns, unit.end_ns, self.next == 0)? {
                self.out.push(Some(record(&au)?));
            }
            self.next += 1;
        }
    }
}

/// The chunk's frame rate, rounded, from its first and last frame times (30 when it can't tell). Only rate
/// control reads it; every unit keeps its own times.
fn frame_rate(units: &[Unit]) -> u32 {
    let (Some(first), Some(last)) = (units.first(), units.last()) else { return 30 };
    let span = last.start_ns.saturating_sub(first.start_ns);
    if units.len() < 2 || span == 0 {
        return 30;
    }
    let rate = ((units.len() - 1) as f64 * 1e9 / span as f64).round();
    if rate.is_finite() { (rate as u32).clamp(1, 120) } else { 30 }
}

fn record(au: &crate::encode::AccessUnit) -> Result<Vec<u8>, TierError> {
    gop::encode(&[Unit { start_ns: au.start_ns, end_ns: au.end_ns, key: au.key, bytes: au.bytes.clone() }])
}
