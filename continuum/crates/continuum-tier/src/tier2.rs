//! Reading frames out of a raw camera's video chunks (docs/CONTINUUM-STORAGE.md), behind the `ffmpeg`
//! feature.
//!
//! [`read_frames`] decodes only the GOPs that overlap a range and returns the frames in it. It reads a raw camera's
//! Tier 1 ([`ChunkFormat::H264Au`], one access unit per record), each GOP assembled from its keyframe's record
//! ([`for_each_gop`]), and the same chunk after the opt-in archive pass made it AV1 ([`ChunkFormat::Av1Au`]).

use std::path::Path;

use crate::TierError;
use crate::chunk::{ChunkFormat, ChunkReader};
use crate::gop::{self, Unit};
use crate::video::{DecodedFrame, codec_of, decode_units};

/// The frames of a raw camera's [`ChunkFormat::H264Au`] chunk whose time overlaps `[start_ns, end_ns)`, decoding only
/// the GOPs that do.
///
/// # Errors
///
/// [`TierError::Codec`] for a chunk that isn't [`ChunkFormat::H264Au`] or a GOP that won't decode; chunk and I/O
/// errors as they come.
pub fn read_frames(path: &Path, start_ns: u64, end_ns: u64) -> Result<Vec<DecodedFrame>, TierError> {
    let mut out = Vec::new();
    for_each_frame(path, start_ns, end_ns, &mut |f| {
        out.push(f);
        Ok(())
    })?;
    Ok(out)
}

/// As [`read_frames`], but each frame goes to `each` as soon as its GOP is decoded, so at most one GOP's frames are
/// held at a time (an export of a long range).
///
/// # Errors
///
/// As [`read_frames`], and whatever `each` returns.
pub fn for_each_frame(
    path: &Path,
    start_ns: u64,
    end_ns: u64,
    each: &mut dyn FnMut(DecodedFrame) -> Result<(), TierError>,
) -> Result<(), TierError> {
    let mut reader = ChunkReader::open(path)?;
    let codec = codec_of(reader.format())
        .ok_or_else(|| TierError::Codec(format!("a {:?} chunk isn't video access units", reader.format())))?;
    for_each_gop(&mut reader, start_ns, end_ns, &mut |units| {
        for f in decode_units(codec, &units)? {
            let end = f.end_ns.max(f.start_ns.saturating_add(1));
            if f.start_ns < end_ns && end > start_ns {
                each(f)?;
            }
        }
        Ok(())
    })
}

/// Each GOP of a raw camera's [`ChunkFormat::H264Au`] chunk that overlaps `[start_ns, end_ns)`, in order, as its
/// access units, each starting with a keyframe. A run retains the entire reference tail through the next IDR or chunk
/// end when presentation times reorder. Monotone no-B-frame streams retain their range-end stop.
///
/// The chunk holds one access unit per record, so a GOP is the run from a keyframe's record to the next keyframe's:
/// the run the range's first unit belongs to is found by walking back to its keyframe. A chunk beginning mid-GOP
/// waits for a keyframe. A purge's tombstone breaks a run: later units cannot decode, so they are skipped up to the
/// next keyframe.
///
/// # Errors
///
/// Chunk and I/O errors, and [`TierError::Codec`] for a chunk that isn't [`ChunkFormat::H264Au`] or a record that
/// isn't a unit record.
pub(crate) fn for_each_gop(
    reader: &mut ChunkReader,
    start_ns: u64,
    end_ns: u64,
    each: &mut dyn FnMut(Vec<Unit>) -> Result<(), TierError>,
) -> Result<(), TierError> {
    if !matches!(reader.format(), ChunkFormat::H264Au | ChunkFormat::Av1Au) {
        return Err(TierError::Codec(format!("a {:?} chunk isn't video access units", reader.format())));
    }
    let records = reader.records().to_vec();
    let reordered = records.windows(2).any(|pair| pair[1].start_ns < pair[0].start_ns);
    let Some(first) = records.iter().position(|r| !r.cut && r.start_ns < end_ns && r.end_exclusive() > start_ns) else {
        return Ok(());
    };
    let mut begin = first;
    while begin > 0 && !records[begin - 1].cut && !gop::decode(&reader.read(begin)?)?.first().is_some_and(|u| u.key) {
        begin -= 1;
    }
    let overlaps =
        |units: &[Unit]| units.iter().any(|u| u.start_ns < end_ns && u.end_ns.max(u.start_ns + 1) > start_ns);
    let mut run: Vec<Unit> = Vec::new();
    let mut broken = true; // until a keyframe starts a run
    for (index, record) in records.iter().enumerate().skip(begin) {
        if !reordered && record.start_ns >= end_ns {
            break; // Existing no-B-frame streams need no later reference tail.
        }
        if record.cut {
            if overlaps(&run) {
                each(std::mem::take(&mut run))?;
            }
            run.clear();
            broken = true;
            continue;
        }
        let units = gop::decode(&reader.read(index)?)?;
        if units.first().is_some_and(|u| u.key) {
            if overlaps(&run) {
                each(std::mem::take(&mut run))?;
            }
            run.clear();
            if units.first().is_some_and(|unit| unit.start_ns >= end_ns) {
                break; // An IDR starts a closed GOP; the preceding GOP's full reference tail was included.
            }
            broken = false;
        }
        if !broken {
            run.extend(units);
        }
    }
    if overlaps(&run) {
        each(run)?;
    }
    Ok(())
}
