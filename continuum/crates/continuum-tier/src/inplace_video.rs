//! H.264 MP4 files registered as Tier 1 chunks in place, and read back by GOP: an imported recording is indexed
//! where it lies. Ingest is the caller (docs/INGEST.md).
//!
//! A directory holds MP4 files (with their `moov`; B-frames are fine) and an `index.jsonl` with each file's exact
//! start (`{"cam", "file", "start", "fps"}`). A file becomes one header-only payload and one chunk on the stream
//! named for its `cam` (`<cam>.video_v1`, encoded video, shape `H264`), the same way [`crate::inplace_audio`]
//! registers audio: deterministic ids, idempotent, the file never moved or rewritten. The chunk's frame index has
//! one frame per GOP: its earliest presented frame's time, and the byte range of its samples in the file's `mdat`
//! ([`crate::mp4`]), so a time range maps to GOPs without opening the file. A file still being written (no `moov`)
//! is skipped with a reason.
//!
//! Reading back, by the GOP's place in the frame index:
//! - [`gop_units`] gives a GOP's access units in decode order as Annex B (the parameter sets on the keyframe), each
//!   with its presentation and decode times, for the gateway's player. It reads the `moov` and that GOP's bytes,
//!   never the whole file.
//! - [`read_frames`] (the `ffmpeg` feature) decodes the frames in a time range to NV12, as
//!   [`crate::tier2::read_frames`] does for a raw camera's chunks, for the vision workers' coverage.
//!
//! The files are stored upright (Ingest turns a rotated picture when it converts it): readers mustn't flip them.
//!
//! The files have closed GOPs (libx264's default, which Ingest's conversion uses), so a GOP decodes on its own. In
//! an open-GOP file, a GOP's leading B-frames refer back to the GOP before, and decoding that GOP alone conceals
//! them; and a sync sample may be a recovery-point I-frame rather than an IDR.
//! Purges follow in-place audio's rule (docs/CONTINUUM-BROKER.md): cut from the catalog, the files left alone.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use continuum_client::{ChunkFrame, Client, StreamSpec, StreamStorage};
use continuum_core::schema::continuum as fb;
use continuum_core::{PayloadFormat, StreamId};

use crate::TierError;
use crate::inplace_audio::{
    Absorbed, InPlace, InPlaceError, Registration, ids_seeded, register_in_place, seconds_to_ns,
};
use crate::mp4::{Mp4Video, read_video, to_annex_b};

/// The shape name of a stream of MP4 files registered in place.
pub const SHAPE: &str = "H264";

/// A track time in the file's timescale as nanoseconds after its start (negative times, before the first edit, as
/// 0).
fn track_ns(t: i64, timescale: u32) -> u64 {
    let t = u128::try_from(t.max(0)).unwrap_or(0);
    u64::try_from(t * 1_000_000_000 / u128::from(timescale)).unwrap_or(u64::MAX)
}

/// `start_ns` plus a signed track time `t`, in UTC nanoseconds (0 at the earliest).
fn on_clock(start_ns: u64, t: i64, timescale: u32) -> u64 {
    let ns = i128::from(t) * 1_000_000_000 / i128::from(timescale);
    u64::try_from((i128::from(start_ns) + ns).max(0)).unwrap_or(u64::MAX)
}

/// The chunk frame index for a recording that starts at `start_ns`: one frame per GOP.
#[must_use]
pub fn gop_frames(video: &Mp4Video, start_ns: u64) -> Vec<ChunkFrame> {
    video
        .gops()
        .iter()
        .map(|g| ChunkFrame {
            start_ns: start_ns.saturating_add(track_ns(g.pts, video.timescale)),
            offset: g.offset,
            length: g.length,
            cut: false,
        })
        .collect()
}

/// The stream a camera's MP4 files are registered on: `<cam>.video_v1`, a sensor at layer 1, encoded H.264 at `fps`.
/// Nothing is ever written to its Tier 0 storage; the smallest slab class is named because one has to be.
#[must_use]
pub fn video_stream_spec(cam: &str, fps: f32) -> StreamSpec {
    StreamSpec {
        name: format!("{cam}.video_v1"),
        kind: fb::StreamKind::Sensor,
        format: fb::PayloadFormat::EncodedVideo,
        shape_name: SHAPE.to_owned(),
        shape_version: 1,
        nominal_rate_hz: fps,
        storage: StreamStorage::Slab { class: 0 },
        anchor: None,
        layer: fb::DomainLayer::Layer1Raw,
    }
}

/// Registers one MP4 file, which starts at `start_ns`, as a payload and a Tier 1 chunk on `stream` (registered
/// by this session with [`video_stream_spec`]). The session must have registered a worker, since only a worker
/// commits chunks. Idempotent; the file's canonical path is its uri and part of its ids. A file never finished,
/// or one that isn't a readable H.264 MP4, is skipped with a reason.
///
/// # Errors
///
/// [`InPlaceError::Tier`] for an I/O error, [`InPlaceError::Client`] when the broker refuses, and [`InPlaceError::Refused`] when
/// the catalog holds something else at the file's time.
pub async fn register_mp4(
    client: &Client,
    stream: StreamId,
    path: &Path,
    start_ns: u64,
) -> Result<Registration, InPlaceError> {
    let path = std::fs::canonicalize(path).map_err(TierError::from)?;
    let mut file = std::fs::File::open(&path).map_err(TierError::from)?;
    let length = file.metadata().map_err(TierError::from)?.len();
    let video = match read_video(&mut file) {
        Ok(v) => v,
        Err(TierError::Codec(why)) => return Ok(Registration::Skipped(format!("{}: {why}", path.display()))),
        Err(e) => return Err(e.into()),
    };
    if !matches!(&video.codec, b"avc1" | b"avc3") {
        return Ok(Registration::Skipped(format!("{}: not H.264", path.display())));
    }
    let frames = gop_frames(&video, start_ns);
    if frames.is_empty() {
        return Ok(Registration::Skipped(format!("{}: no keyframe", path.display())));
    }
    // A chunk holds at most this many frames, so a file with more GOPs (all-intra, say) would be refused after its
    // payloads were ingested, leaving them behind: it's skipped before anything is sent.
    if frames.len() > continuum_catalog::MAX_CHUNK_FRAMES {
        return Ok(Registration::Skipped(format!(
            "{}: {} GOPs, more than a chunk's {} frames",
            path.display(),
            frames.len(),
            continuum_catalog::MAX_CHUNK_FRAMES
        )));
    }
    let end_ns = start_ns.saturating_add(track_ns(video.duration(), video.timescale)).max(start_ns.saturating_add(1));
    let samples = video.samples.len() as u64;
    let (_, chunk_id) = ids_seeded("video", &path, samples, start_ns);
    // One payload per GOP (the catalog's chunk absorbs one per frame), spanning its frames' presentation times.
    let payloads = video
        .gops()
        .iter()
        .zip(&frames)
        .enumerate()
        .map(|(i, (g, f))| Absorbed {
            // Seeded by the GOP's index too, so two GOPs that start at one time still get their own ids.
            id: ids_seeded(&format!("video gop {i}"), &path, samples, f.start_ns).0,
            span: (
                f.start_ns,
                start_ns.saturating_add(track_ns(g.pts_end, video.timescale)).max(f.start_ns.saturating_add(1)),
            ),
            seq: u32::try_from(i).unwrap_or(u32::MAX),
        })
        .collect();
    let place = InPlace {
        stream,
        span: (start_ns, end_ns),
        format: PayloadFormat::EncodedVideo,
        shape_name: SHAPE.to_owned(),
        frames,
        payloads,
    };
    register_in_place(client, &path, &place, chunk_id, length, "in-place video").await
}

/// One line of a directory's `index.jsonl`.
#[derive(serde::Deserialize)]
struct IndexLine {
    cam: String,
    file: String,
    start: f64,
    #[serde(default)]
    fps: Option<f32>,
}

/// Registers every file of camera `cam` that the `index.jsonl` in `video_dir` lists, on
/// `<cam>.video_v1` (registered here). A file missing from disk, unfinished, or not H.264 is skipped with a reason;
/// a file listed twice registers once, at its last start. A line that isn't JSON is an error.
///
/// # Errors
///
/// [`InPlaceError::Index`] for an unreadable index or line, and as [`register_mp4`].
pub async fn register_video_index(
    client: &Client,
    cam: &str,
    video_dir: &Path,
) -> Result<Vec<(PathBuf, Registration)>, InPlaceError> {
    let index_path = video_dir.join("index.jsonl");
    let text = std::fs::read_to_string(&index_path)
        .map_err(|e| InPlaceError::Index(format!("{}: {e}", index_path.display())))?;
    let mut lines: Vec<(PathBuf, u64)> = Vec::new();
    let mut at = std::collections::HashMap::new();
    let mut fps = None;
    for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let l: IndexLine = serde_json::from_str(line)
            .map_err(|e| InPlaceError::Index(format!("{} line {}: {e}", index_path.display(), n + 1)))?;
        if l.cam != cam {
            continue;
        }
        let start_ns = seconds_to_ns(l.start).ok_or_else(|| {
            InPlaceError::Index(format!("{} line {}: start {}", index_path.display(), n + 1, l.start))
        })?;
        fps = fps.or(l.fps);
        let path = video_dir.join(&l.file);
        match at.get(&path) {
            Some(&i) => lines[i] = (path, start_ns),
            None => {
                at.insert(path.clone(), lines.len());
                lines.push((path, start_ns));
            }
        }
    }
    let mut stream = None;
    let mut out = Vec::with_capacity(lines.len());
    for (path, start_ns) in lines {
        if !path.is_file() {
            out.push((path.clone(), Registration::Skipped(format!("{}: not on disk", path.display()))));
            continue;
        }
        let id = match stream {
            Some(id) => id,
            None => {
                let registered = client.register_stream(&video_stream_spec(cam, fps.unwrap_or(0.0))).await?;
                *stream.insert(StreamId(registered.stream_id))
            }
        };
        let r = register_mp4(client, id, &path, start_ns).await?;
        out.push((path, r));
    }
    Ok(out)
}

/// One access unit of an MP4 file registered in place.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GopUnit {
    /// Annex B; the keyframe's begins with the SPS and PPS.
    pub bytes: Vec<u8>,
    /// Its presentation time, UTC nanoseconds: what a decoder's output frame carries.
    pub pts_ns: u64,
    /// Its presentation end.
    pub end_ns: u64,
    /// Its decode time, UTC nanoseconds (before `pts_ns` for a frame a B-frame refers back to).
    pub dts_ns: u64,
    /// The IDR (the GOP's sync sample): decoding can start here.
    pub key: bool,
}

/// The access units of GOP `gop` (its place in the chunk's frame index) of the recording at `path`, which starts at
/// `start_ns` (the payload's start), in decode order.
///
/// # Errors
///
/// [`TierError::Codec`] for a file that isn't a readable MP4 or has no such GOP, and an I/O error as it comes.
pub fn gop_units(path: &Path, start_ns: u64, gop: usize) -> Result<Vec<GopUnit>, TierError> {
    let mut file = std::fs::File::open(path)?;
    let video = read_video(&mut file)?;
    units_of(&mut file, &video, start_ns, gop)
}

fn units_of(file: &mut std::fs::File, video: &Mp4Video, start_ns: u64, gop: usize) -> Result<Vec<GopUnit>, TierError> {
    let g = *video
        .gops()
        .get(gop)
        .ok_or_else(|| TierError::Codec(format!("no GOP {gop} in a recording of {}", video.gops().len())))?;
    if g.length > 256 << 20 {
        return Err(TierError::Codec(format!("a GOP of {} bytes", g.length)));
    }
    let mut bytes = vec![0u8; usize::try_from(g.length).map_err(|_| TierError::Codec("GOP too big".to_owned()))?];
    file.seek(SeekFrom::Start(g.offset))?;
    file.read_exact(&mut bytes)?;
    video.samples[g.first..g.first + g.count]
        .iter()
        .map(|s| {
            let at = usize::try_from(s.offset - g.offset).map_err(|_| TierError::Codec("offset".to_owned()))?;
            let raw = bytes
                .get(at..at + s.size as usize)
                .ok_or_else(|| TierError::Codec("a sample outside its GOP's bytes".to_owned()))?;
            let pts = start_ns.saturating_add(track_ns(s.pts, video.timescale));
            Ok(GopUnit {
                bytes: to_annex_b(video, raw, s.key)?,
                pts_ns: pts,
                end_ns: pts.saturating_add(track_ns(i64::from(s.duration), video.timescale)),
                // On the presentation clock, so a frame that B-frames refer back to decodes before it's shown.
                dts_ns: on_clock(
                    start_ns,
                    i64::try_from(s.dts).unwrap_or(i64::MAX) - video.media_start,
                    video.timescale,
                ),
                key: s.key,
            })
        })
        .collect()
}

/// The frames of the recording at `path` (which starts at `start_ns`) whose time overlaps `[from, to)`, decoded to
/// NV12 and in presentation order, as [`crate::tier2::read_frames`] gives a raw camera's.
///
/// # Errors
///
/// As [`gop_units`], and a decoder failure.
#[cfg(feature = "ffmpeg")]
pub fn read_frames(
    path: &Path,
    start_ns: u64,
    from: u64,
    to: u64,
) -> Result<Vec<crate::video::DecodedFrame>, TierError> {
    let mut file = std::fs::File::open(path)?;
    let video = read_video(&mut file)?;
    let frames = gop_frames(&video, start_ns);
    let ends: Vec<u64> =
        video.gops().iter().map(|g| start_ns.saturating_add(track_ns(g.pts_end, video.timescale))).collect();
    let mut out = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        if f.start_ns >= to || ends[i] <= from {
            continue;
        }
        let units: Vec<crate::gop::Unit> = units_of(&mut file, &video, start_ns, i)?
            .into_iter()
            .map(|u| crate::gop::Unit { start_ns: u.pts_ns, end_ns: u.end_ns, key: u.key, bytes: u.bytes })
            .collect();
        out.extend(crate::video::decode_units(ffmpeg_next::codec::Id::H264, &units)?.into_iter().filter(|f| {
            let end = f.end_ns.max(f.start_ns.saturating_add(1));
            f.start_ns < to && end > from
        }));
    }
    Ok(out)
}
