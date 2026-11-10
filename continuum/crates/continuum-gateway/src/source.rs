//! Recorded frames straight from the tier worker's chunk files, for a time range, until the broker's `Query` is
//! wired (then the catalog says where every frame is; this module keeps reading the bytes).
//!
//! A chunk file (`continuum-tier`'s container) holds one stream's payloads as the producer wrote them. The gateway
//! finds a stream's chunks by their file names (`<stream>-<created_ns>-<n>.cnk`), reads the records whose time
//! overlaps the range, and sniffs what each payload is: a JPEG frame (`FF D8`), an Annex B access unit (a start
//! code), or an Arrow IPC table. Random access requires an actual H264 IDR or HEVC IRAP picture, not just
//! parameter sets. Predictive H264 records retain decode order and complete selected reference chains even
//! when PTS regresses. The Tier0 broker path uses at most 60 seconds of context on each side and 4096 items;
//! insufficient/truncated context reports unavailable. Each GOP is capped at 4096 units/64 MiB and a selected
//! context at 8192 units/128 MiB. Raw LZ4 and FLAC have no browser decoder and are skipped and counted.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use continuum_tier::gop;
use continuum_tier::{ChunkError, ChunkFormat, ChunkReader};

use crate::header::{FrameHeader, PayloadType};

/// What the source found for one stream.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StreamSummary {
    /// The stream's id (the broker's `u32`).
    pub id: u32,
    /// Its registered name, when the broker is in the loop (empty from the chunk directory alone).
    #[serde(default)]
    pub name: String,
    /// A producer is connected for it now (always false from the chunk directory alone).
    #[serde(default)]
    pub live: bool,
    /// How its chunks store payloads (`passthrough`, `lz4`, `flac`, `arrow`, `h264_au`, `opus`); for a stream the
    /// broker lists but no chunk holds yet, its registered shape name in lower case (`mjpeg`, say).
    pub format: String,
    /// The earliest record, microseconds UTC.
    pub start_us: u64,
    /// Past the latest record, microseconds UTC.
    pub end_us: u64,
    /// Chunk files found.
    pub chunks: usize,
    /// Records across them.
    pub frames: usize,
}

/// One frame ready to send: its header and the bytes after it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    /// The frame's header.
    pub header: FrameHeader,
    /// The payload.
    pub bytes: Vec<u8>,
    /// What the sniff found: `avc`, `hevc`, `jpeg`, or `arrow`.
    pub codec: &'static str,
    /// Metadata from the exact logical chunk whose bytes were read.
    pub media: Option<std::sync::Arc<crate::media::MediaContext>>,
}

/// What a range read produced.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RangeRead {
    /// The frames, in time order by record; a raw camera's access units come in decode order (B-frames reorder, so
    /// a player sorts them by time once decoded).
    pub frames: Vec<Frame>,
    /// Records in the range the browser can't decode (raw, FLAC, or unrecognised bytes).
    pub skipped: usize,
    /// The range held more than `max` frames: `frames` is the first `max`, and `next_us` says where to continue.
    pub next_us: Option<u64>,
}

/// Why the source couldn't read.
#[derive(Debug, thiserror::Error)]
pub enum SourceError {
    /// The chunk directory couldn't be listed.
    #[error("listing {0}: {1}")]
    List(PathBuf, std::io::Error),
    /// A chunk couldn't be read.
    #[error(transparent)]
    Chunk(#[from] ChunkError),
    /// A predictive range cannot be reconstructed within its bounded context.
    #[error("H264 range unavailable: {0}")]
    Video(&'static str),
}

/// A physical access unit with its native decode sequence. Missing bytes break references.
pub(crate) struct AuEntry {
    pub seq: u64,
    pub start_ns: u64,
    pub end_ns: u64,
    pub frame: Option<Frame>,
}

/// Assemble complete reference chains in physical decode order, never presentation order.
/// A selected GOP includes its prefix and tail even when only a B-picture overlaps the range.
/// One GOP is bounded by 4096 units / 64 MiB; exceeding context is unavailable, not a truncated delta chain.
pub(crate) fn select_h264_gops(
    entries: impl IntoIterator<Item = Result<AuEntry, SourceError>>,
    start_ns: u64,
    end_ns: u64,
    budget: usize,
    max_bytes: usize,
    tail_complete: bool,
) -> Result<(Vec<(u64, Frame)>, usize), SourceError> {
    let mut out = Vec::new();
    let mut group: Vec<Frame> = Vec::new();
    let mut overlap = false;
    let mut bytes = 0usize;
    let mut total_bytes = 0usize;
    let mut skipped = 0;
    let mut previous = None;
    let mut missing_prefix = false;
    let flush = |group: &mut Vec<Frame>,
                 overlap: &mut bool,
                 bytes: &mut usize,
                 out: &mut Vec<(u64, Frame)>,
                 total_bytes: &mut usize| {
        let mut stop = false;
        if *overlap && !group.is_empty() {
            let key = group.iter().map(|f| f.header.capture_us.saturating_mul(1000)).min().expect("nonempty GOP");
            stop = (out.len() >= budget || *total_bytes >= max_bytes)
                && key / 1000 > start_ns / 1000
                && out.iter().map(|(_, f)| f.header.capture_us).max().is_some_and(|frontier| key / 1000 > frontier);
            out.extend(group.drain(..).map(|frame| (key, frame)));
            *total_bytes += *bytes;
        } else {
            group.clear();
        }
        *overlap = false;
        *bytes = 0;
        stop
    };
    for entry in entries {
        let entry = entry?;
        let hit = entry.start_ns < end_ns && entry.end_ns.max(entry.start_ns.saturating_add(1)) > start_ns;
        // Seq 0 on both sides is no sequence at all (a raw Mac camera's Tier 1 before 2026-11-05 wrote 0 on every
        // record): the units are in decode order, so they aren't a gap. Without this, Review played only keyframes.
        let gap = previous.is_some_and(|seq: u64| {
            !(seq == 0 && entry.seq == 0)
                && entry.seq != seq.wrapping_add(1)
                && entry.seq != u64::from((seq as u32).wrapping_add(1))
                && !(entry.seq <= seq && entry.frame.as_ref().is_some_and(|f| f.header.keyframe))
        });
        previous = Some(entry.seq);
        if entry.frame.is_none() || gap {
            if overlap || hit && !entry.frame.as_ref().is_some_and(|f| f.header.keyframe) {
                skipped += 1;
            }
            if flush(&mut group, &mut overlap, &mut bytes, &mut out, &mut total_bytes) {
                return Ok((out, skipped));
            }
            if entry.frame.is_none() {
                continue;
            }
        }
        let Some(frame) = entry.frame else { continue };
        if frame.header.keyframe && flush(&mut group, &mut overlap, &mut bytes, &mut out, &mut total_bytes) {
            return Ok((out, skipped));
        }
        if group.is_empty() && !frame.header.keyframe {
            missing_prefix |= hit;
            continue;
        }
        bytes = bytes.saturating_add(frame.bytes.len());
        if group.len() >= 4096 || bytes > 64 * 1024 * 1024 {
            return Err(SourceError::Video("GOP exceeds the bounded decode context"));
        }
        overlap |= hit;
        group.push(frame);
        if out.len().saturating_add(group.len()) > 8192 || total_bytes.saturating_add(bytes) > 128 * 1024 * 1024 {
            return Err(SourceError::Video("selected GOPs exceed the bounded decode context"));
        }
    }
    if overlap && !tail_complete {
        return Err(SourceError::Video("complete GOP tail is outside the query context"));
    }
    flush(&mut group, &mut overlap, &mut bytes, &mut out, &mut total_bytes);
    if missing_prefix {
        skipped += 1;
    }
    Ok((out, skipped))
}

/// One chunk file's part of a stream summary, kept so a listing reopens only files that changed.
#[derive(Clone, Debug)]
struct FileSummary {
    id: u32,
    format: String,
    first_us: u64,
    last_us: u64,
    frames: usize,
}

/// Per-file summaries by path, with the size and modification time they were read at, and the last full listing.
/// A listing used to open every chunk file on every `streams` request: with 173k files (a Mac, 2026-11-02) that
/// took minutes, the Live tab's request never got an answer, and no camera showed.
#[derive(Debug, Default)]
struct Summaries {
    /// Each chunk's size and modification time when read, its summary, and, for a chunk still being written (no
    /// footer yet), the offset its next records start at: a live chunk grows on every listing, and re-reading it from
    /// the start took about 1.8 s with six cameras (#2772), so only its new tail is read.
    files: Files,
    last: Option<(std::time::Instant, Vec<StreamSummary>)>,
}

/// Each chunk's cache entry (see [`Summaries`]). Summaries are shared, so carrying ~300k of them from one listing to
/// the next copies pointers, not strings (#2772).
type Files = std::collections::HashMap<PathBuf, (u64, std::time::SystemTime, Option<Arc<FileSummary>>, Option<u64>)>;

/// How long one listing answers again before the directory is looked at anew.
const LISTING_REUSE: std::time::Duration = std::time::Duration::from_secs(5);

/// What one chunk directory held when last read (#2908): its modification time then, when it was read, and its
/// chunk files by stream id.
#[derive(Debug)]
struct Listed {
    modified: std::time::SystemTime,
    read_at: std::time::SystemTime,
    files: Arc<Vec<(u32, PathBuf)>>,
}

/// A directory whose modification time is this close to when it was last read is read again: a file added in the
/// same tick of a coarse clock wouldn't move it.
const MTIME_SLACK: std::time::Duration = std::time::Duration::from_secs(2);

/// A directory of chunk files.
#[derive(Clone, Debug)]
pub struct ChunkSource {
    dir: PathBuf,
    summaries: Arc<std::sync::Mutex<Summaries>>,
    /// Each chunk directory's files as last read (see `continuum_tier::layout`): one whose modification time hasn't
    /// moved since isn't read again, so a listing reads only the hours still being written (#2908).
    listed: Arc<std::sync::Mutex<std::collections::HashMap<PathBuf, Listed>>>,
    /// A full listing has been made. Outside the lock, which a listing holds while it runs (the first one, minutes
    /// over a large directory): asking must never wait on it.
    ready: Arc<std::sync::atomic::AtomicBool>,
    /// Tests: how long a listing takes, to show a slow one holds nothing else back (#2772).
    #[cfg(test)]
    listing_delay: Option<std::time::Duration>,
}

impl ChunkSource {
    /// The chunks under `dir` (which may not exist yet: then every stream is empty).
    #[must_use]
    pub fn new(dir: &Path) -> Self {
        Self {
            dir: dir.to_path_buf(),
            summaries: Arc::default(),
            listed: Arc::default(),
            ready: Arc::default(),
            #[cfg(test)]
            listing_delay: None,
        }
    }

    /// Tests: a source whose every listing takes `delay` (a live chunk re-read whole used to take about 1.8 s).
    #[cfg(test)]
    #[must_use]
    pub(crate) fn slow_listing(dir: &Path, delay: std::time::Duration) -> Self {
        Self { listing_delay: Some(delay), ..Self::new(dir) }
    }

    /// Whether a full listing has been made, so `streams` answers from the cache.
    #[must_use]
    pub fn summaries_ready(&self) -> bool {
        self.ready.load(std::sync::atomic::Ordering::Acquire)
    }

    /// The directory.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn h264_frames(
        &self,
        stream: u32,
        start_ns: u64,
        end_ns: u64,
        max: usize,
        max_bytes: usize,
    ) -> Result<RangeRead, SourceError> {
        let mut metadata = Vec::new();
        let mut paths = Vec::new();
        let mut others = Vec::new();
        let mut epoch = 0u64;
        let mut epoch_anchor = None;
        let mut previous_seq = None;
        let mut files = self.chunk_files(Some(stream))?;
        files.sort_by_key(|(_, path)| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .and_then(|s| s.split('-').nth(1))
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0)
        });
        for (_, path) in files {
            let mut reader = match ChunkReader::open_in(&path, start_ns, end_ns) {
                Ok(Some(reader)) => reader,
                Ok(None) | Err(ChunkError::NotAChunk(_)) => continue,
                Err(ChunkError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e.into()),
            };
            if !is_au(reader.format()) {
                others.push(path);
                continue;
            }
            let anchor = decode_anchor(reader.records());
            let first = reader.records().iter().min_by_key(|r| r.seq.wrapping_sub(anchor)).copied();
            let first_key = first.is_some_and(|r| {
                !r.cut
                    && reader
                        .read(reader.records().iter().position(|x| x.seq == r.seq).expect("record"))
                        .is_ok_and(|b| starts_at_key(&b))
            });
            if first_key && first.is_some_and(|r| previous_seq.is_some_and(|last| r.seq <= last)) {
                epoch += 1;
                epoch_anchor = None;
            }
            let epoch_start = *epoch_anchor.get_or_insert(anchor);
            previous_seq = reader.records().iter().max_by_key(|r| r.seq.wrapping_sub(anchor)).map(|r| r.seq);
            let ri = paths.len();
            for (index, record) in reader.records().iter().enumerate() {
                metadata.push((epoch, record.seq.wrapping_sub(epoch_start), ri, index));
            }
            paths.push(path);
        }
        metadata.sort_unstable();
        let mut current: Option<(usize, ChunkReader)> = None;
        let entries = metadata.into_iter().map(|(_, _, ri, index)| {
            if current.as_ref().is_none_or(|(open, _)| *open != ri) {
                current = Some((ri, ChunkReader::open(&paths[ri])?));
            }
            au_entry(&mut current.as_mut().expect("opened chunk").1, index, stream)
        });
        let (mut frames, mut skipped) = select_h264_gops(entries, start_ns, end_ns, max, max_bytes, true)?;
        let mut other_frames = Vec::new();
        for path in others {
            let (more, lost) = Self::chunk_frames(&path, stream, start_ns, end_ns, max.saturating_add(1))?;
            if frames.len().saturating_add(other_frames.len()).saturating_add(more.len()) > 8192
                || frames
                    .iter()
                    .chain(other_frames.iter())
                    .chain(more.iter())
                    .map(|(_, f)| f.bytes.len())
                    .sum::<usize>()
                    > 128 * 1024 * 1024
            {
                return Err(SourceError::Video("mixed records exceed the bounded decode context"));
            }
            other_frames.extend(more);
            skipped += lost;
        }
        frames = merge_record_groups(frames, other_frames);
        Ok(limit_gops(frames, skipped, max, max_bytes, start_ns))
    }

    fn chunk_files(&self, stream: Option<u32>) -> Result<Vec<(u32, PathBuf)>, SourceError> {
        let list = |e: std::io::Error, at: &Path| SourceError::List(at.to_path_buf(), e);
        let dirs = continuum_tier::layout::chunk_dirs(&self.dir, stream.map(continuum_core::StreamId))
            .map_err(|e| list(e, &self.dir))?;
        let mut out = Vec::new();
        for dir in &dirs {
            for (id, path) in self.files_in(dir).map_err(|e| list(e, dir))?.iter() {
                if stream.is_none_or(|s| s == *id) {
                    out.push((*id, path.clone()));
                }
            }
        }
        if stream.is_none() {
            // A directory gone since (an hour aged out whole) leaves the cache.
            let present: std::collections::HashSet<&PathBuf> = dirs.iter().collect();
            self.listed.lock().unwrap_or_else(std::sync::PoisonError::into_inner).retain(|d, _| present.contains(d));
        }
        out.sort();
        Ok(out)
    }

    /// One chunk directory's chunk files by stream id: as last read while its modification time hasn't moved (and
    /// isn't within MTIME_SLACK of that read), else read again.
    fn files_in(&self, dir: &Path) -> std::io::Result<Arc<Vec<(u32, PathBuf)>>> {
        let modified = match fs::metadata(dir) {
            Ok(meta) => meta.modified().unwrap_or(std::time::UNIX_EPOCH),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Arc::default()),
            Err(e) => return Err(e),
        };
        if let Some(seen) = self.listed.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get(dir)
            && seen.modified == modified
            && modified + MTIME_SLACK < seen.read_at
        {
            return Ok(seen.files.clone());
        }
        let read_at = std::time::SystemTime::now();
        let mut files = Vec::new();
        for path in continuum_tier::layout::files_in(dir)? {
            if path.extension().and_then(|e| e.to_str()) != Some("cnk") {
                continue;
            }
            let id = path.file_name().and_then(|n| n.to_str()).and_then(|n| n.split('-').next());
            if let Some(id) = id.and_then(|s| s.parse::<u32>().ok()) {
                files.push((id, path));
            }
        }
        let files = Arc::new(files);
        self.listed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(dir.to_path_buf(), Listed { modified, read_at, files: files.clone() });
        Ok(files)
    }

    /// The time each chunk of `stream` covers, nanoseconds `[start, end)`, for the chunks overlapping
    /// `[start_ns, end_ns)`, in no particular order.
    ///
    /// # Errors
    ///
    /// [`SourceError`] when the directory or a chunk can't be read.
    pub fn chunk_spans(&self, stream: u32, start_ns: u64, end_ns: u64) -> Result<Vec<(u64, u64)>, SourceError> {
        let mut out = Vec::new();
        for (_, path) in self.chunk_files(Some(stream))? {
            let reader = match ChunkReader::open_in(&path, start_ns, end_ns) {
                Ok(Some(r)) => r,
                Ok(None) => continue,
                Err(ChunkError::NotAChunk(_)) => continue,
                Err(ChunkError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e.into()),
            };
            let live = reader.records().iter().filter(|r| !r.cut);
            let first = live.clone().map(|r| r.start_ns).min();
            let last = live.map(continuum_tier::Record::end_exclusive).max();
            if let (Some(a), Some(b)) = (first, last)
                && a < end_ns
                && b > start_ns
            {
                out.push((a, b));
            }
        }
        Ok(out)
    }

    /// Every stream with chunks, by id. Each file is read once; later listings reopen only files whose size or
    /// modification time changed, and a listing under LISTING_REUSE old is answered again.
    ///
    /// # Errors
    ///
    /// [`SourceError`] when the directory or a chunk can't be read.
    pub fn streams(&self) -> Result<Vec<StreamSummary>, SourceError> {
        // Held for the whole listing, so one runs at a time and the map moves from one to the next without a copy;
        // a listing that waited for it answers from the fresh `last`.
        let mut cache = self.summaries.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((at, listing)) = &cache.last
            && at.elapsed() < LISTING_REUSE
        {
            return Ok(listing.clone());
        }
        #[cfg(test)]
        if let Some(delay) = self.listing_delay {
            std::thread::sleep(delay); // the listing's work, under the lock as a real one's is
        }
        let mut known = std::mem::take(&mut cache.files);
        let mut files = Files::with_capacity(known.len());
        let by_id = match self.list_into(&mut known, &mut files) {
            Ok(by_id) => by_id,
            Err(e) => {
                // Keep what's known, so the next listing doesn't read every chunk again.
                known.extend(files);
                cache.files = known;
                return Err(e);
            }
        };
        let listing: Vec<StreamSummary> = by_id.into_values().collect();
        cache.files = files; // chunks gone from the directory are left behind in `known`
        cache.last = Some((std::time::Instant::now(), listing.clone()));
        self.ready.store(true, std::sync::atomic::Ordering::Release);
        Ok(listing)
    }

    /// One listing's pass over the directory: each chunk's entry moved from `known` (or read) into `files`, and the
    /// streams they add up to.
    fn list_into(&self, known: &mut Files, files: &mut Files) -> Result<BTreeMap<u32, StreamSummary>, SourceError> {
        let mut by_id: BTreeMap<u32, StreamSummary> = BTreeMap::new();
        for (id, path) in self.chunk_files(None)? {
            let entry = match known.remove(&path) {
                // A finished chunk (a summary, and no footer-less tail to follow) doesn't change: a purge keeps its
                // records' places and times. So it isn't stat'ed again: the directory holds ~300k chunks, and
                // stat'ing each on every listing took about 2 s (#2772). Only new and still-growing chunks are.
                Some(finished @ (_, _, Some(_), None)) => finished,
                cached => {
                    let Ok(meta) = fs::metadata(&path) else { continue };
                    let (len, modified) = (meta.len(), meta.modified().unwrap_or(std::time::UNIX_EPOCH));
                    let (summary, resume) = match cached {
                        Some((l, m, sum, at)) if l == len && m == modified => (sum, at),
                        // Grown since, and still being written when last read: read only what was added.
                        Some((l, _, Some(sum), Some(at))) if len > l => Self::grow(&sum, &path, at)?,
                        _ => Self::summarize(id, &path)?,
                    };
                    (len, modified, summary, resume)
                }
            };
            let summary = entry.2.clone();
            files.insert(path, entry);
            let Some(f) = summary else { continue };
            let s = by_id.entry(f.id).or_insert_with(|| StreamSummary {
                id: f.id,
                name: String::new(),
                live: false,
                format: f.format.clone(),
                start_us: f.first_us,
                end_us: f.last_us,
                chunks: 0,
                frames: 0,
            });
            s.start_us = s.start_us.min(f.first_us);
            s.end_us = s.end_us.max(f.last_us);
            s.chunks += 1;
            s.frames += f.frames;
        }
        Ok(by_id)
    }

    /// One chunk's summary, read whole (`None` for a file that isn't a chunk, is gone, or holds no records), and where its next records will start when it's still being written (no
    /// footer): the end of its last whole record.
    fn summarize(id: u32, path: &Path) -> Result<(Option<Arc<FileSummary>>, Option<u64>), SourceError> {
        let reader = match ChunkReader::open(path) {
            Ok(r) => r,
            Err(ChunkError::NotAChunk(_)) => return Ok((None, None)),
            Err(ChunkError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => return Ok((None, None)),
            Err(e) => return Err(e.into()),
        };
        let (Some(f), Some(l)) = (reader.records().first(), reader.records().last()) else { return Ok((None, None)) };
        let resume =
            (!reader.finalized()).then(|| reader.records().iter().map(|r| r.offset + u64::from(r.len)).max()).flatten();
        Ok((
            Some(Arc::new(FileSummary {
                id,
                format: format_name(reader.format()).to_owned(),
                first_us: reader.records().iter().map(|r| r.start_ns).min().unwrap_or(f.start_ns) / 1_000,
                last_us: reader
                    .records()
                    .iter()
                    .map(continuum_tier::Record::end_exclusive)
                    .max()
                    .unwrap_or(l.end_exclusive())
                    .div_ceil(1_000),
                frames: reader.records().len(),
            })),
            resume,
        ))
    }

    /// A chunk still being written, summarized before up to `at`: its summary with the records added since.
    fn grow(
        before: &FileSummary,
        path: &Path,
        at: u64,
    ) -> Result<(Option<Arc<FileSummary>>, Option<u64>), SourceError> {
        let (added, next) = match continuum_tier::scan_tail(path, at) {
            Ok(found) => found,
            Err(ChunkError::NotAChunk(_)) => return Ok((None, None)),
            Err(ChunkError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => return Ok((None, None)),
            Err(e) => return Err(e.into()),
        };
        let mut sum = (*before).clone();
        for r in &added {
            sum.first_us = sum.first_us.min(r.start_ns / 1_000);
            sum.last_us = sum.last_us.max(r.end_exclusive().div_ceil(1_000));
        }
        sum.frames += added.len();
        Ok((Some(Arc::new(sum)), next)) // no offset once it's finished: then it's left alone, like any finished chunk
    }

    /// The frames in `[start_ns, end_ns)` of one chunk file, keyed by the record's start (a GOP's units share their
    /// record's key, so a time sort keeps their decode order), for a range the broker located in chunks. Reading
    /// stops once more than `budget` frames are in hand and the next record starts a new microsecond, so a request
    /// never reads far past its cap (the caller cuts and continues from `next_us`).
    ///
    /// # Errors
    ///
    /// [`SourceError`] when the chunk can't be opened or read (a vanished file is the caller's to skip).
    pub fn chunk_frames(
        path: &Path,
        stream: u32,
        start_ns: u64,
        end_ns: u64,
        budget: usize,
    ) -> Result<(Vec<(u64, Frame)>, usize), SourceError> {
        Self::chunk_frames_snapshot(path, stream, start_ns, end_ns, budget, None)
    }

    /// As [`ChunkSource::chunk_frames`], using the catalog snapshot's authoritative frame cuts when given.
    /// # Errors
    /// Chunk I/O or codec errors.
    pub fn chunk_frames_snapshot(
        path: &Path,
        stream: u32,
        start_ns: u64,
        end_ns: u64,
        budget: usize,
        snapshot: Option<&[continuum_client::ChunkFrame]>,
    ) -> Result<(Vec<(u64, Frame)>, usize), SourceError> {
        let Some(mut reader) = ChunkReader::open_in(path, start_ns, end_ns)? else {
            return Ok((Vec::new(), 0));
        };
        if is_au(reader.format()) {
            return h264_reader_frames(&mut reader, stream, start_ns, end_ns, budget, snapshot);
        }
        // A raw camera's tombstones are kept as markers: the units after one can't decode until the next keyframe.
        let au = is_au(reader.format());
        let wanted: Vec<(u64, usize, bool)> = reader
            .records()
            .iter()
            .enumerate()
            .filter(|(_, r)| (au || !r.cut) && r.start_ns >= start_ns && r.start_ns < end_ns)
            .map(|(i, r)| (r.start_ns, i, r.cut || !snapshot_allows(snapshot, r.start_ns)))
            .collect();
        let mut out = Vec::new();
        let mut skipped = 0;
        let mut codec: Option<&'static str> = None;
        let mut last_key: Option<u64> = None;
        let mut broken = false;
        for (start, index, cut) in wanted {
            if cut {
                broken = true;
                continue;
            }
            if out.len() > budget && last_key.is_some_and(|k| k / 1_000 != start / 1_000) {
                break;
            }
            let bytes = match reader.read(index) {
                Ok(b) => b,
                Err(ChunkError::Corrupt { .. } | ChunkError::NoSuchFrame(_)) => {
                    skipped += 1;
                    continue;
                }
                Err(e) => return Err(e.into()),
            };
            match record_frames(reader.format(), stream, start, bytes, &mut codec, reader.flip()) {
                Some(frames) if au && !resumes(&mut broken, &frames) => {}
                Some(frames) => out.extend(frames.into_iter().map(|f| (start, f))),
                None => skipped += 1,
            }
            last_key = Some(start);
        }
        Ok((out, skipped))
    }

    /// The frames of `stream` whose start is in `[start_ns, end_ns)`, in time order, at most `max` of them and
    /// about `max_bytes` of payload. A cut never splits records that share a start microsecond, so `next_us` is
    /// always past everything sent and a client continuing from it makes progress. A raw camera's Tier 1 video
    /// (H264Au, #900) keeps one access unit per record, served in decode order (each with its own time and keyframe
    /// flag), since a unit decodes only after the ones before it: a page is cut only before a GOP cursor; any overlapping GOP retains its reference prefix and tail in decode order.
    /// Missing/corrupt references suppress subsequent deltas until the next IDR and increment skipped. Opus
    /// records are skipped (no audio path yet). A chunk that vanishes between the listing and the open (a purge) is
    /// skipped; a record whose CRC fails is counted in `skipped`.
    ///
    /// # Errors
    ///
    /// [`SourceError`] when the directory can't be listed or a chunk fails for a reason other than being gone.
    pub fn frames(
        &self,
        stream: u32,
        start_ns: u64,
        end_ns: u64,
        max: usize,
        max_bytes: usize,
    ) -> Result<RangeRead, SourceError> {
        // Keep only metadata while sorting across chunks, so a small range never holds a reader per historical file.
        let mut chunks = Vec::new();
        let mut wanted: Vec<(u64, u32, usize, usize, bool)> = Vec::new();
        for (_, path) in self.chunk_files(Some(stream))? {
            let reader = match ChunkReader::open_in(&path, start_ns, end_ns) {
                Ok(Some(reader)) => reader,
                Ok(None) | Err(ChunkError::NotAChunk(_)) => continue,
                Err(ChunkError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e.into()),
            };
            let ri = chunks.len();
            let format = reader.format();
            let au = is_au(format);
            for (index, rec) in reader.records().iter().enumerate() {
                if (au || !rec.cut) && rec.start_ns >= start_ns && rec.start_ns < end_ns {
                    wanted.push((rec.start_ns, rec.seq, ri, index, rec.cut));
                }
            }
            chunks.push((path, format));
        }
        if chunks.iter().any(|(_, format)| is_au(*format)) {
            return self.h264_frames(stream, start_ns, end_ns, max, max_bytes);
        }
        wanted.sort_unstable();
        let mut out = RangeRead::default();
        let mut sent_bytes = 0usize;
        let mut codec: Option<&'static str> = None;
        let mut keyed = false; // an H264Au range has reached its first keyframe
        let mut current: Option<(usize, ChunkReader)> = None;
        for (i, &(start, _, ri, index, cut)) in wanted.iter().enumerate() {
            if cut {
                keyed = false; // a purge broke the GOP: wait for the next keyframe, as at a range's start
                continue;
            }
            let full = out.frames.len() >= max || sent_bytes >= max_bytes;
            let same_us = i > 0 && wanted[i - 1].0 / 1_000 == start / 1_000;
            let au = is_au(chunks[ri].1);
            if full && !same_us && !au {
                out.next_us = Some(start / 1_000);
                break;
            }
            if current.as_ref().is_none_or(|(opened, _)| *opened != ri) {
                // Drop the previous archive before opening another, even when record order interleaves files.
                drop(current.take());
                let reader = match ChunkReader::open_in(&chunks[ri].0, start_ns, end_ns) {
                    Ok(Some(reader)) => reader,
                    Ok(None) | Err(ChunkError::NotAChunk(_)) => continue,
                    Err(ChunkError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => continue,
                    Err(e) => return Err(e.into()),
                };
                current = Some((ri, reader));
            }
            let reader = &mut current.as_mut().expect("opened current chunk").1;
            if reader.records().get(index).is_some_and(|record| record.cut) {
                if au {
                    keyed = false; // a purge may have rewritten the file since metadata discovery
                }
                continue;
            }
            let bytes = match reader.read(index) {
                Ok(b) => b,
                Err(ChunkError::Corrupt { .. } | ChunkError::NoSuchFrame(_)) => {
                    out.skipped += 1;
                    continue;
                }
                Err(e) => return Err(e.into()),
            };
            if au {
                let key = starts_at_key(&bytes);
                if full && !same_us && key {
                    out.next_us = Some(start / 1_000); // cut only before a keyframe, so the next page decodes
                    break;
                }
                if !keyed && !key {
                    continue; // before the range's first keyframe: undecodable
                }
                keyed = true;
            }
            match record_frames(reader.format(), stream, start, bytes, &mut codec, reader.flip()) {
                Some(frames) => {
                    for f in frames {
                        sent_bytes += f.bytes.len();
                        out.frames.push(f);
                    }
                }
                None => out.skipped += 1,
            }
        }
        Ok(out)
    }
}
// Sequence numbers span less than half the u32 ring within a bounded chunk. The largest circular gap is
// outside the captured sequence, so its successor anchors decode order even across u32::MAX -> 0.
fn decode_anchor(records: &[continuum_tier::Record]) -> u32 {
    let mut seqs: Vec<_> = records.iter().map(|r| r.seq).collect();
    seqs.sort_unstable();
    seqs.dedup();
    let Some(&first) = seqs.first() else { return 0 };
    seqs.iter()
        .copied()
        .zip(seqs.iter().copied().cycle().skip(1))
        .take(seqs.len())
        .max_by_key(|(a, b)| b.wrapping_sub(*a))
        .map_or(first, |(_, b)| b)
}

fn au_entry(reader: &mut ChunkReader, index: usize, stream: u32) -> Result<AuEntry, SourceError> {
    let record = reader.records()[index];
    let frame = if record.cut {
        None
    } else {
        match reader.read(index) {
            Ok(bytes) => record_frames(reader.format(), stream, record.start_ns, bytes, &mut None, reader.flip())
                .and_then(|mut f| f.pop()),
            Err(ChunkError::Corrupt { .. } | ChunkError::NoSuchFrame(_)) => None,
            Err(e) => return Err(e.into()),
        }
    };
    Ok(AuEntry { seq: u64::from(record.seq), start_ns: record.start_ns, end_ns: record.end_ns, frame })
}

fn snapshot_allows(snapshot: Option<&[continuum_client::ChunkFrame]>, start_ns: u64) -> bool {
    snapshot.is_none_or(|frames| {
        frames.binary_search_by_key(&start_ns, |f| f.start_ns).ok().is_some_and(|i| !frames[i].cut)
    })
}

fn h264_reader_frames(
    reader: &mut ChunkReader,
    stream: u32,
    start_ns: u64,
    end_ns: u64,
    budget: usize,
    snapshot: Option<&[continuum_client::ChunkFrame]>,
) -> Result<(Vec<(u64, Frame)>, usize), SourceError> {
    let anchor = decode_anchor(reader.records());
    let mut order: Vec<_> = reader.records().iter().enumerate().map(|(i, r)| (r.seq.wrapping_sub(anchor), i)).collect();
    order.sort_unstable();
    select_h264_gops(
        order.into_iter().map(|(_, i)| {
            let record = &reader.records()[i];
            if !snapshot_allows(snapshot, record.start_ns) {
                return Ok(AuEntry {
                    seq: u64::from(record.seq),
                    start_ns: record.start_ns,
                    end_ns: record.end_ns,
                    frame: None,
                });
            }
            au_entry(reader, i, stream)
        }),
        start_ns,
        end_ns,
        budget,
        usize::MAX,
        true,
    )
}

/// Insert independent legacy record groups without sorting the native access units or GOPs by PTS.
pub(crate) fn merge_record_groups(native: Vec<(u64, Frame)>, other: Vec<(u64, Frame)>) -> Vec<(u64, Frame)> {
    let mut other = other.into_iter().peekable();
    let mut out = Vec::with_capacity(native.len() + other.len());
    let mut previous = None;
    for (key, frame) in native {
        if previous != Some(key) {
            while other.peek().is_some_and(|(other_key, _)| *other_key < key) {
                out.push(other.next().expect("peeked record"));
            }
        }
        out.push((key, frame));
        previous = Some(key);
    }
    out.extend(other);
    out
}

pub(crate) fn limit_gops(
    frames: Vec<(u64, Frame)>,
    skipped: usize,
    max: usize,
    max_bytes: usize,
    start_ns: u64,
) -> RangeRead {
    let mut out = RangeRead { skipped, ..Default::default() };
    let mut sent_bytes = 0usize;
    let mut last = None;
    for (key, frame) in frames {
        if (out.frames.len() >= max || sent_bytes >= max_bytes)
            && key / 1000 > start_ns / 1000
            && last.is_some_and(|last| key / 1000 > last)
        {
            out.next_us = Some(key / 1000);
            break;
        }
        sent_bytes += frame.bytes.len();
        last = Some(last.map_or(frame.header.capture_us, |previous: u64| previous.max(frame.header.capture_us)));
        out.frames.push(frame);
    }
    out
}

/// Whether a raw camera's access unit is served: not while a tombstone has broken its GOP (`broken`), until a
/// keyframe starts the next one.
fn resumes(broken: &mut bool, frames: &[Frame]) -> bool {
    if frames.first().is_some_and(|f| f.header.keyframe) {
        *broken = false;
    }
    !*broken
}

/// Whether a raw camera's unit record starts with a keyframe (a record that doesn't decode counts as no).
/// A chunk of one access unit per record ([`gop`]): a raw camera's H.264, or the same chunk after the archive pass
/// made it AV1 (#1479). Both keep each unit's keyframe flag and decode order the same way.
fn is_au(format: ChunkFormat) -> bool {
    matches!(format, ChunkFormat::H264Au | ChunkFormat::Av1Au)
}

fn starts_at_key(record: &[u8]) -> bool {
    continuum_tier::gop::decode(record).is_ok_and(|units| units.first().is_some_and(|u| u.key))
}

/// The frames one record yields: a raw camera's unit record's access unit, else the record itself as the sniff
/// classifies it; `None` for bytes the browser can't decode. `codec` carries the stream's video codec between
/// records of one read. `flip` is the captured camera orientation, never a current device setting; only picture
/// payloads carry it, because overlays are already in upright vision coordinates.
pub fn record_frames(
    format: ChunkFormat,
    stream: u32,
    start_ns: u64,
    bytes: Vec<u8>,
    codec: &mut Option<&'static str>,
    flip: bool,
) -> Option<Vec<Frame>> {
    // A raw camera's Tier 1 access unit (H264Au, #900) is a one-unit record ([`gop`]), and H.264 ("avc", the name
    // the sniff uses); after the opt-in archive pass (Av1Au, #1479) the same record holds an AV1 temporal unit.
    if is_au(format) {
        let name = if format == ChunkFormat::Av1Au { "av1" } else { "avc" };
        let units = gop::decode(&bytes).ok()?;
        *codec = Some(name);
        return Some(
            units
                .into_iter()
                .map(|u| Frame {
                    header: FrameHeader { flip, ..FrameHeader::at_ns(stream, PayloadType::Video, u.key, u.start_ns) },
                    bytes: u.bytes,
                    codec: name,
                    media: None,
                })
                .collect(),
        );
    }
    let (payload_type, keyframe, name) = classify(format, &bytes, *codec)?;
    if payload_type == PayloadType::Video {
        *codec = Some(name);
    }
    Some(vec![Frame {
        header: FrameHeader {
            flip: flip && matches!(payload_type, PayloadType::Video | PayloadType::Jpeg),
            ..FrameHeader::at_ns(stream, payload_type, keyframe, start_ns)
        },
        bytes,
        codec: name,
        media: None,
    }])
}

fn format_name(f: ChunkFormat) -> &'static str {
    match f {
        ChunkFormat::Passthrough => "passthrough",
        ChunkFormat::Lz4 => "lz4",
        ChunkFormat::Flac => "flac",
        ChunkFormat::ArrowIpc => "arrow",
        ChunkFormat::H264Au => "h264_au",
        ChunkFormat::Av1Au => "av1_au",
        ChunkFormat::Opus => "opus",
        ChunkFormat::PcmBatch => "pcm_batch",
        ChunkFormat::Mp4 => "mp4",
    }
}

/// What a record's bytes are, as the browser needs to know it: the payload type, whether it's a keyframe, and the
/// codec's name. `known` is the stream's video codec as decided by an earlier access unit of the same read, for one
/// that carries no parameter sets. `None` for bytes a browser can't decode (or a video access unit whose codec
/// can't be told yet).
#[must_use]
pub fn classify(
    format: ChunkFormat,
    bytes: &[u8],
    known: Option<&'static str>,
) -> Option<(PayloadType, bool, &'static str)> {
    match format {
        ChunkFormat::ArrowIpc => Some((PayloadType::Arrow, false, "arrow")),
        // MP4s registered in place are served through continuum_tier::inplace_video::gop_units (wired with the Review player).
        ChunkFormat::Lz4 | ChunkFormat::Flac | ChunkFormat::PcmBatch | ChunkFormat::Mp4 => None,
        // A raw camera's unit record is unpacked by `record_frames` before this is asked; an Opus record (a second of
        // packets) has no player path yet.
        ChunkFormat::H264Au | ChunkFormat::Av1Au | ChunkFormat::Opus => None,
        ChunkFormat::Passthrough => {
            if bytes.starts_with(&[0xff, 0xd8]) {
                return Some((PayloadType::Jpeg, true, "jpeg"));
            }
            let (codec, keyframe) = annexb_codec(bytes, known)?;
            Some((PayloadType::Video, keyframe, codec))
        }
    }
}

/// H.264 NAL type 5, an IDR picture.
const H264_IDR: u8 = 5;
/// HEVC NAL types 16 to 21 are IRAP pictures; 33 is the SPS.
const HEVC_IRAP: std::ops::RangeInclusive<u8> = 16..=21;
const HEVC_SPS: u8 = 33;

/// For an Annex B access unit: `avc` or `hevc`, and whether it's a keyframe. The codec is told from its parameter
/// sets, which is unambiguous: an H.264 SPS header byte (`0x27`, `0x47`, or `0x67`: type 7 with a nonzero
/// `nal_ref_idc`) reads as a reserved type in HEVC, and an HEVC SPS (`0x42 0x01`: type 33, layer 0, temporal id
/// 0) reads in H.264 as the long-dead data partition A. An access unit without parameter sets takes `known`, the
/// codec of an earlier unit of the same stream; with neither it cannot be decoded and is `None`.
/// Only an actual IDR (H.264) or IRAP (HEVC) picture is a keyframe; parameter sets alone are not.
/// `None` when the bytes do not start with a start code.
#[must_use]
pub fn annexb_codec(bytes: &[u8], known: Option<&'static str>) -> Option<(&'static str, bool)> {
    if !(bytes.starts_with(&[0, 0, 1]) || bytes.starts_with(&[0, 0, 0, 1])) {
        return None;
    }
    let units = nal_units(bytes);
    let avc_ps = units.iter().any(|n| matches!(n.first(), Some(0x27 | 0x47 | 0x67)));
    let hevc_ps = units.iter().any(|n| n.first() == Some(&(HEVC_SPS << 1)) && n.get(1) == Some(&0x01));
    let codec = if avc_ps {
        "avc"
    } else if hevc_ps {
        "hevc"
    } else {
        known?
    };
    let keyframe = units.iter().any(|n| {
        let Some(&b0) = n.first() else { return false };
        if codec == "avc" {
            b0 & 0x80 == 0 && b0 & 0x1f == H264_IDR
        } else {
            b0 & 0x80 == 0 && HEVC_IRAP.contains(&((b0 >> 1) & 0x3f)) && n.get(1) == Some(&0x01)
        }
    });
    Some((codec, keyframe))
}
/// The NAL units of an Annex B byte stream, without their start codes.
fn nal_units(bytes: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= bytes.len() {
        if bytes[i] == 0 && bytes[i + 1] == 0 && bytes[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::with_capacity(starts.len());
    for (k, &s) in starts.iter().enumerate() {
        let mut end = starts.get(k + 1).map_or(bytes.len(), |&n| n - 3);
        while end > s && bytes[end - 1] == 0 {
            end -= 1;
        }
        out.push(&bytes[s..end]);
    }
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use continuum_core::StreamId;
    use continuum_tier::ChunkWriter;

    use super::*;

    /// Where a test's chunk file `name` (`<stream>-<created_ns>-<n>.cnk`) goes: its stream's hour directory
    /// (`continuum_tier::layout`), made. A name that doesn't parse goes in stream 0's first hour.
    pub(crate) fn cnk(dir: &Path, name: &str) -> PathBuf {
        let mut parts = name.split('-');
        let stream = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
        let created = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
        let hour = continuum_tier::layout::hour_dir(dir, StreamId(stream), created);
        continuum_tier::layout::make_hour_dir(dir, &hour).expect("hour");
        hour.join(name)
    }

    fn write(dir: &Path, stream: u32, name: &str, format: ChunkFormat, frames: &[(u64, Vec<u8>)]) {
        let mut w = ChunkWriter::create(&cnk(dir, name), StreamId(stream), format, 1).expect("create");
        for (i, (t, bytes)) in frames.iter().enumerate() {
            w.append(*t, *t, if is_au(format) { (*t / 1_000_000) as u32 } else { i as u32 }, bytes).expect("append");
        }
        w.close().expect("close");
    }

    fn native_chunk(dir: &Path, filename: &str, created: u64, records: &[(u32, u64, bool)]) {
        let mut writer =
            ChunkWriter::create(&cnk(dir, filename), StreamId(8), ChunkFormat::H264Au, created).expect("create");
        for &(seq, pts, key) in records {
            let bytes = if key { vec![0, 0, 1, 0x67, 0x4d, 0, 0, 1, 0x65, 0x88] } else { vec![0, 0, 1, 0x41, 0x88] };
            let unit = gop::Unit { start_ns: pts, end_ns: pts, key, bytes };
            writer.append(pts, pts, seq, &gop::encode(&[unit]).expect("unit")).expect("append");
        }
        writer.close().expect("close");
    }

    #[test]
    fn records_without_a_sequence_still_play_their_p_frames() {
        let dir = tempfile::tempdir().expect("tempdir");
        native_chunk(
            dir.path(),
            "8-100-1.cnk",
            100,
            &[(0, 100_000, true), (0, 200_000, false), (0, 300_000, false), (0, 400_000, false)],
        );
        let source = ChunkSource::new(dir.path());
        let all = source.frames(8, 0, 500_000, 100, usize::MAX).expect("range");
        assert_eq!(all.frames.iter().map(|f| f.header.capture_us).collect::<Vec<_>>(), [100, 200, 300, 400]);
        assert_eq!(all.skipped, 0);
    }

    #[test]
    fn native_b_frames_keep_decode_order_prefix_tail_and_gop_pagination() {
        let dir = tempfile::tempdir().expect("tempdir");
        native_chunk(
            dir.path(),
            "8-100-1.cnk",
            100,
            &[
                (1, 100_000, true),
                (2, 300_000, false),
                (3, 200_000, false),
                (4, 400_000, true),
                (5, 600_000, false),
                (6, 500_000, false),
            ],
        );
        let source = ChunkSource::new(dir.path());
        let seek = source.frames(8, 200_000, 201_000, 1, usize::MAX).expect("B seek");
        assert_eq!(seek.frames.iter().map(|f| f.header.capture_us).collect::<Vec<_>>(), [100, 300, 200]);
        assert_eq!(seek.next_us, None);
        let first = source.frames(8, 0, 700_000, 1, usize::MAX).expect("first");
        assert_eq!(first.frames.iter().map(|f| f.header.capture_us).collect::<Vec<_>>(), [100, 300, 200]);
        assert_eq!(first.next_us, Some(400));
        let second = source.frames(8, 400_000, 700_000, 1, usize::MAX).expect("second");
        assert_eq!(second.frames.iter().map(|f| f.header.capture_us).collect::<Vec<_>>(), [400, 600, 500]);
        let (group, skipped) =
            ChunkSource::chunk_frames(&cnk(dir.path(), "8-100-1.cnk"), 8, 200_000, 201_000, 1).expect("chunk");
        assert_eq!(skipped, 0);
        assert_eq!(
            group.iter().map(|(key, f)| (*key, f.header.capture_us)).collect::<Vec<_>>(),
            [(100_000, 100), (100_000, 300), (100_000, 200)]
        );
    }

    #[test]
    fn overlapping_gop_pts_spans_are_coalesced_before_cursor_advances() {
        let dir = tempfile::tempdir().expect("tempdir");
        native_chunk(
            dir.path(),
            "8-1-1.cnk",
            1,
            &[(1, 100_000, true), (2, 300_000, false), (3, 250_000, true), (4, 450_000, false), (5, 500_000, true)],
        );
        let src = ChunkSource::new(dir.path());
        let first = src.frames(8, 0, 600_000, 1, usize::MAX).expect("overlap");
        assert_eq!(first.frames.iter().map(|f| f.header.capture_us).collect::<Vec<_>>(), [100, 300, 250, 450]);
        assert_eq!(first.next_us, Some(500));
        let next = src.frames(8, 500_000, 600_000, 1, usize::MAX).expect("next");
        assert_eq!(next.frames.iter().map(|f| f.header.capture_us).collect::<Vec<_>>(), [500]);
    }

    #[test]
    fn regressing_gop_minima_do_not_reorder_native_decode_groups() {
        let dir = tempfile::tempdir().expect("tempdir");
        native_chunk(
            dir.path(),
            "8-1-1.cnk",
            1,
            &[(1, 100_000, true), (2, 200_000, false), (3, 300_000, true), (4, 50_000, false), (5, 400_000, true)],
        );
        let read = ChunkSource::new(dir.path()).frames(8, 0, 500_000, 1, usize::MAX).expect("regressing GOP");
        assert_eq!(read.frames.iter().map(|f| f.header.capture_us).collect::<Vec<_>>(), [100, 200, 300, 50]);
        assert_eq!(read.next_us, Some(400));
    }

    #[test]
    fn native_sequence_u32_wrap_does_not_reorder_or_break_a_gop() {
        let dir = tempfile::tempdir().expect("tempdir");
        native_chunk(
            dir.path(),
            "8-100-1.cnk",
            100,
            &[(u32::MAX - 1, 100_000, true), (u32::MAX, 300_000, false), (0, 200_000, false)],
        );
        let read = ChunkSource::new(dir.path()).frames(8, 0, 400_000, 100, usize::MAX).expect("wrapped");
        assert_eq!(read.frames.iter().map(|f| f.header.capture_us).collect::<Vec<_>>(), [100, 300, 200]);
        assert_eq!(read.skipped, 0);
    }

    #[test]
    fn reconnect_sequence_reset_keeps_old_epoch_before_new_epoch() {
        let dir = tempfile::tempdir().expect("tempdir");
        native_chunk(dir.path(), "8-100-1.cnk", 100, &[(1000, 100_000, true), (1001, 200_000, false)]);
        native_chunk(dir.path(), "8-300-2.cnk", 300, &[(1, 300_000, true), (2, 400_000, false)]);
        let read = ChunkSource::new(dir.path()).frames(8, 0, 500_000, 100, usize::MAX).expect("epochs");
        assert_eq!(read.frames.iter().map(|f| f.header.capture_us).collect::<Vec<_>>(), [100, 200, 300, 400]);
        assert_eq!(read.skipped, 0);
    }

    #[test]
    fn gops_sharing_cursor_microsecond_are_never_split_or_repeated() {
        let dir = tempfile::tempdir().expect("tempdir");
        native_chunk(
            dir.path(),
            "8-1-1.cnk",
            1,
            &[
                (1, 1_000_100, true),
                (2, 1_000_150, false),
                (3, 1_000_200, true),
                (4, 1_000_250, false),
                (5, 2_000_000, true),
            ],
        );
        let src = ChunkSource::new(dir.path());
        let first = src.frames(8, 1_000_000, 3_000_000, 1, usize::MAX).expect("colliding GOPs");
        assert_eq!(first.frames.len(), 4);
        assert_eq!(first.next_us, Some(2000));
        let next = src.frames(8, 2_000_000, 3_000_000, 1, usize::MAX).expect("next");
        assert_eq!(next.frames.len(), 1);
    }

    #[test]
    fn pathological_reference_chains_and_colliding_cursors_have_hard_unit_bounds() {
        let entries = |count: usize, all_keys: bool| {
            (0..count).map(move |i| {
                Ok(AuEntry {
                    seq: i as u64,
                    start_ns: 0,
                    end_ns: 0,
                    frame: Some(Frame {
                        header: FrameHeader::at_ns(8, PayloadType::Video, all_keys || i == 0, 0),
                        bytes: vec![0],
                        codec: "avc",
                        media: None,
                    }),
                })
            })
        };
        assert!(matches!(
            select_h264_gops(entries(4097, false), 0, 1, 1, usize::MAX, true),
            Err(SourceError::Video("GOP exceeds the bounded decode context"))
        ));
        assert!(matches!(
            select_h264_gops(entries(8193, true), 0, 1, 1, usize::MAX, true),
            Err(SourceError::Video("selected GOPs exceed the bounded decode context"))
        ));
    }

    #[test]
    fn parameter_sets_alone_never_claim_random_access() {
        assert_eq!(annexb_codec(&[0, 0, 1, 0x67, 0x4d, 0, 0, 1, 0x68, 0x88], None), Some(("avc", false)));
        assert_eq!(annexb_codec(&[0, 0, 1, 0x42, 0x01, 0x88], None), Some(("hevc", false)));
    }

    #[test]
    fn jpeg_orientation_is_kept_but_arrow_tables_never_rotate() {
        let mut codec = None;
        let jpeg = vec![0xff, 0xd8, 0xff, 0xd9];
        let frames = record_frames(ChunkFormat::Passthrough, 7, 1_000, jpeg.clone(), &mut codec, true).expect("jpeg");
        assert!(frames[0].header.flip);
        assert_eq!(frames[0].bytes, jpeg, "the gateway forwards native pixels without transcoding");
        let tables = record_frames(ChunkFormat::ArrowIpc, 7, 1_000, vec![1], &mut codec, true).expect("table");
        assert!(!tables[0].header.flip);
    }

    #[test]
    fn every_video_access_unit_keeps_the_records_captured_orientation() {
        let units = vec![
            gop::Unit { start_ns: 10_000, end_ns: 20_000, key: true, bytes: vec![1] },
            gop::Unit { start_ns: 20_000, end_ns: 30_000, key: false, bytes: vec![2] },
        ];
        let record = gop::encode(&units).expect("GOP");
        let frames = record_frames(ChunkFormat::H264Au, 7, 10_000, record, &mut None, true).expect("video");
        assert_eq!(frames.len(), 2);
        assert!(frames.iter().all(|frame| frame.header.flip));
        assert!(frames[0].header.keyframe && !frames[1].header.keyframe);
    }

    #[test]
    fn archived_chunk_orientation_reaches_directory_and_catalog_readers() {
        for flip in [false, true] {
            let dir = tempfile::tempdir().expect("tempdir");
            let path = cnk(dir.path(), "7-1-1.cnk");
            let mut writer = ChunkWriter::create(&path, StreamId(7), ChunkFormat::Passthrough, 1).expect("create");
            writer.set_flip(flip).expect("orientation");
            writer.append(1_000, 1_000, 0, &[0xff, 0xd8, 0xff, 0xd9]).expect("append");
            writer.close().expect("close");
            let read = ChunkSource::new(dir.path()).frames(7, 0, 2_000, 10, 1 << 20).expect("directory read");
            assert_eq!(read.frames.len(), 1);
            assert_eq!(read.frames[0].header.flip, flip);
            let (keyed, skipped) = ChunkSource::chunk_frames(&path, 7, 0, 2_000, 10).expect("catalog read");
            assert_eq!(skipped, 0);
            assert_eq!(keyed.len(), 1);
            assert_eq!(keyed[0].1.header.flip, flip);
        }
    }

    #[test]
    fn a_raw_cameras_units_after_a_tombstone_wait_for_the_next_keyframe() {
        // One access unit per record (a one-unit GOP record); unit 2 purged, so 3 can't decode and 4 (a key) resumes.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = cnk(dir.path(), "5-1-1.cnk");
        let id = |k: u8| continuum_core::Uuid([k; 16]);
        let mut w = ChunkWriter::create(&path, StreamId(5), ChunkFormat::H264Au, 1).expect("create");
        for k in 0..6u8 {
            let t = u64::from(k) * 1_000_000;
            let unit = gop::Unit { start_ns: t, end_ns: t, key: k % 4 == 0, bytes: vec![0, 0, 1, 0x65, k] };
            w.append_payload(id(k), t, t, u32::from(k), &gop::encode(&[unit]).expect("encode")).expect("append");
        }
        w.close().expect("close");
        let cut = std::collections::BTreeSet::from([id(2)]);
        continuum_tier::service::purge_file(&path, &[], &cut, false).expect("purge");

        let read = ChunkSource::new(dir.path()).frames(5, 0, 10_000_000, 100, 1 << 20).expect("frames");
        let got: Vec<_> = read.frames.iter().map(|f| (f.header.capture_us, f.header.keyframe, f.codec)).collect();
        assert_eq!(got, [(0, true, "avc"), (1_000, false, "avc"), (4_000, true, "avc"), (5_000, false, "avc")]);
        assert_eq!(read.skipped, 2, "missing references are explicit degraded coverage");
        let (frames, skipped) = ChunkSource::chunk_frames(&path, 5, 0, 10_000_000, 100).expect("chunk frames");
        let at: Vec<u64> = frames.iter().map(|(_, f)| f.header.capture_us).collect();
        assert_eq!((at, skipped), (vec![0, 1_000, 4_000, 5_000], 2));
    }

    #[test]
    fn an_h264au_range_starts_and_is_cut_only_at_a_keyframe() {
        let dir = tempfile::tempdir().expect("tempdir");
        let au = |t: u64, key: bool| {
            let bytes =
                if key { vec![0, 0, 0, 1, 0x67, 0x4d, 0, 0, 0, 1, 0x65, 0x88] } else { vec![0, 0, 1, 0x41, 0x9a] };
            (t, gop::encode(&[gop::Unit { start_ns: t, end_ns: t, key, bytes }]).expect("encode"))
        };
        // Two GOPs of three units: a keyframe at 1 ms and at 4 ms.
        let units: Vec<(u64, Vec<u8>)> = [(1, true), (2, false), (3, false), (4, true), (5, false), (6, false)]
            .map(|(ms, k)| au(ms * 1_000_000, k))
            .into();
        write(dir.path(), 8, "8-1-1.cnk", ChunkFormat::H264Au, &units);
        let src = ChunkSource::new(dir.path());
        let us = |r: &RangeRead| r.frames.iter().map(|f| (f.header.capture_us, f.header.keyframe)).collect::<Vec<_>>();

        // A seek into the first GOP retains the reference prefix and complete selected tail.
        let mid = src.frames(8, 2_000_000, 10_000_000, 100, usize::MAX).expect("frames");
        assert_eq!(
            us(&mid),
            [(1_000, true), (2_000, false), (3_000, false), (4_000, true), (5_000, false), (6_000, false)]
        );
        // A page full after two units runs to the end of its GOP, then continues at the next keyframe.
        let page = src.frames(8, 0, 10_000_000, 2, usize::MAX).expect("frames");
        assert_eq!(us(&page), [(1_000, true), (2_000, false), (3_000, false)]);
        assert_eq!(page.next_us, Some(4_000));
    }

    #[test]
    fn interleaved_chunk_files_keep_global_order_and_keyframe_continuation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let au = |t, key| {
            (
                t,
                gop::encode(&[gop::Unit {
                    start_ns: t,
                    end_ns: t,
                    key,
                    bytes: vec![0, 0, 1, if key { 0x65 } else { 0x41 }, 0x88],
                }])
                .expect("encode"),
            )
        };
        write(
            dir.path(),
            8,
            "8-1-1.cnk",
            ChunkFormat::H264Au,
            &[au(1_000_000, true), au(3_000_000, false), au(5_000_000, true)],
        );
        write(dir.path(), 8, "8-2-2.cnk", ChunkFormat::H264Au, &[au(2_000_000, false), au(4_000_000, false)]);
        let source = ChunkSource::new(dir.path());
        let page = source.frames(8, 0, 10_000_000, 2, usize::MAX).expect("first page");
        assert_eq!(page.frames.iter().map(|f| f.header.capture_us).collect::<Vec<_>>(), vec![1000, 2000, 3000, 4000]);
        assert!(page.frames.iter().all(|f| f.codec == "avc"));
        assert_eq!(page.next_us, Some(5000));
        let next = source.frames(8, 5_000_000, 10_000_000, 2, usize::MAX).expect("continuation");
        assert_eq!(next.frames.len(), 1);
        assert_eq!(next.frames[0].header.capture_us, 5000);
        assert!(next.frames[0].header.keyframe);
    }

    #[test]
    fn an_archived_av1_chunk_follows_its_h264_one_in_a_range() {
        // Before the archive pass (#1479) the first chunk was H.264 too; now an H.264 chunk is followed by an AV1 one.
        let dir = tempfile::tempdir().expect("tempdir");
        let unit = |t: u64, key: bool, bytes: Vec<u8>| {
            (t, gop::encode(&[gop::Unit { start_ns: t, end_ns: t, key, bytes }]).expect("encode"))
        };
        let h264 = |ms: u64, key: bool| {
            let bytes =
                if key { vec![0, 0, 0, 1, 0x67, 0x4d, 0, 0, 0, 1, 0x65, 0x88] } else { vec![0, 0, 1, 0x41, 0x9a] };
            unit(ms * 1_000_000, key, bytes)
        };
        // A temporal delimiter, then a sequence header on the keyframe; a delta unit has the delimiter only.
        let av1 = |ms: u64, key: bool| {
            let bytes = if key { vec![0x12, 0, 0x0a, 1, 0, 0x32, 1, 0] } else { vec![0x12, 0, 0x32, 1, 0] };
            unit(ms * 1_000_000, key, bytes)
        };
        write(dir.path(), 8, "8-1-1.cnk", ChunkFormat::H264Au, &[h264(1, true), h264(2, false)]);
        write(dir.path(), 8, "8-3-2.cnk", ChunkFormat::Av1Au, &[av1(3, true), av1(4, false), av1(5, false)]);
        let source = ChunkSource::new(dir.path());

        let all = source.frames(8, 0, 10_000_000, 100, usize::MAX).expect("frames");
        let got: Vec<_> = all.frames.iter().map(|f| (f.header.capture_us, f.header.keyframe, f.codec)).collect();
        assert_eq!(
            got,
            [
                (1_000, true, "avc"),
                (2_000, false, "avc"),
                (3_000, true, "av1"),
                (4_000, false, "av1"),
                (5_000, false, "av1")
            ]
        );
        // A seek into the AV1 chunk starts at its keyframe, as an H.264 one does.
        let mid = source.frames(8, 4_000_000, 10_000_000, 100, usize::MAX).expect("frames");
        assert_eq!(mid.frames.iter().map(|f| f.header.capture_us).collect::<Vec<_>>(), [3_000, 4_000, 5_000]);
        assert_eq!(format_name(ChunkFormat::Av1Au), "av1_au");
        assert_eq!(classify(ChunkFormat::Av1Au, &[0x12, 0], None), None, "unpacked by record_frames, never sniffed");
    }

    #[test]
    fn a_raw_cameras_access_unit_record_is_one_h264_frame() {
        let unit = continuum_tier::gop::Unit {
            start_ns: 5_000,
            end_ns: 5_000,
            key: true,
            bytes: vec![0, 0, 0, 1, 0x65, 0x88],
        };
        let record = continuum_tier::gop::encode(&[unit]).expect("encode");
        let mut codec = None;
        let frames = record_frames(ChunkFormat::H264Au, 7, 5_000, record, &mut codec, false).expect("frames");
        assert_eq!(frames.len(), 1);
        assert_eq!((frames[0].codec, frames[0].header.keyframe), ("avc", true));
        assert_eq!(codec, Some("avc"));
        assert_eq!(format_name(ChunkFormat::H264Au), "h264_au");
    }

    fn finished_chunk(path: &Path, stream: u32, t: u64) {
        let mut w = continuum_tier::ChunkWriter::create(path, StreamId(stream), ChunkFormat::Passthrough, t).unwrap();
        w.append(t, t, 0, &[0xff, 0xd8, 0]).unwrap();
        w.close().unwrap();
    }

    fn listed_ids(src: &ChunkSource) -> Vec<(u32, usize)> {
        src.summaries.lock().expect("lock").last = None; // past LISTING_REUSE
        src.streams().expect("streams").iter().map(|s| (s.id, s.chunks)).collect()
    }

    /// Where a test's chunk of `stream` created at `t` goes (`continuum_tier::layout`), its hour directory made.
    fn at(dir: &Path, stream: u32, t: u64, n: u32) -> PathBuf {
        let hour = continuum_tier::layout::hour_dir(dir, StreamId(stream), t);
        continuum_tier::layout::make_hour_dir(dir, &hour).unwrap();
        hour.join(format!("{stream}-{t}-{n}.cnk"))
    }

    /// #2908: each stream's hour directories are listed, and nothing else.
    #[test]
    fn the_hour_directories_chunks_are_listed() {
        let dir = tempfile::tempdir().unwrap();
        finished_chunk(&dir.path().join("4-1-1.cnk"), 4, 1_000_000); // not where chunks go
        finished_chunk(&at(dir.path(), 5, 2_000_000, 1), 5, 2_000_000);
        finished_chunk(&at(dir.path(), 6, 3_000_000, 1), 6, 3_000_000);
        assert_eq!(listed_ids(&ChunkSource::new(dir.path())), vec![(5, 1), (6, 1)]);
    }

    /// #2908: a chunk directory whose modification time hasn't moved isn't read again (a finished hour's), and one
    /// whose time moved is.
    #[test]
    fn an_unchanged_chunk_directory_is_not_read_again() {
        let dir = tempfile::tempdir().unwrap();
        let old = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
        let first = at(dir.path(), 4, 1_000_000, 1);
        let hour = first.parent().unwrap().to_path_buf();
        let set_mtime = |t: std::time::SystemTime| fs::File::open(&hour).unwrap().set_modified(t).unwrap();
        finished_chunk(&first, 4, 1_000_000);
        set_mtime(old);
        let src = ChunkSource::new(dir.path());
        assert_eq!(listed_ids(&src), vec![(4, 1)]);
        finished_chunk(&at(dir.path(), 4, 2_000_000, 2), 4, 2_000_000);
        set_mtime(old); // as if nothing had changed
        assert_eq!(listed_ids(&src), vec![(4, 1)], "the directory wasn't read again");
        set_mtime(std::time::SystemTime::now());
        assert_eq!(listed_ids(&src), vec![(4, 2)], "its time moved: read again");
    }

    #[test]
    fn a_listing_is_cached_and_marks_the_source_ready() {
        let dir = tempfile::tempdir().unwrap();
        let src = ChunkSource::new(dir.path());
        assert!(!src.summaries_ready(), "nothing listed yet");
        assert!(src.streams().expect("streams").is_empty());
        assert!(src.summaries_ready(), "a full listing marks it ready, so sessions use it");
    }

    /// #2772: a live chunk grows between listings. Only its new records are read, and the listing matches one made
    /// by reading it whole.
    #[test]
    fn a_growing_chunk_is_listed_from_its_new_records_alone() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = cnk(dir.path(), "4-1-1.cnk");
        let mut w =
            continuum_tier::ChunkWriter::create(&path, StreamId(4), ChunkFormat::Passthrough, 1).expect("create");
        for i in 0..3u32 {
            let t = 1_000_000 + u64::from(i) * 1_000_000;
            w.append(t, t, i, &[0xff, 0xd8, i as u8]).expect("append");
        }
        w.sync().expect("sync");
        let src = ChunkSource::new(dir.path());
        let listed = |src: &ChunkSource| {
            src.summaries.lock().expect("lock").last = None; // past LISTING_REUSE
            src.streams().expect("streams").iter().map(|s| (s.id, s.frames, s.start_us, s.end_us)).collect::<Vec<_>>()
        };
        assert_eq!(listed(&src), vec![(4, 3, 1_000, 3_001)]);
        let resume = |src: &ChunkSource| src.summaries.lock().expect("lock").files.get(&path).and_then(|f| f.3);
        let before = resume(&src).expect("still being written: where its next records start");
        for i in 3..5u32 {
            let t = 1_000_000 + u64::from(i) * 1_000_000;
            w.append(t, t, i, &[0xff, 0xd8, i as u8]).expect("append");
        }
        w.sync().expect("sync");
        assert_eq!(listed(&src), vec![(4, 5, 1_000, 5_001)]);
        assert!(resume(&src).expect("still growing") > before, "read on from where it stopped");
        assert_eq!(listed(&ChunkSource::new(dir.path())), listed(&src), "the same as reading it whole");
        w.close().expect("close");
        assert_eq!(listed(&src), vec![(4, 5, 1_000, 5_001)], "finished: its footer doesn't count as frames");
        assert_eq!(resume(&src), None, "finished: nothing more to follow, so it isn't looked at again");
    }

    /// #2772: ~300k finished chunks were stat'ed on every listing (about 2 s). A finished one is trusted as read; a new
    /// one is still found, and a deleted one still goes.
    #[test]
    fn finished_chunks_are_listed_without_a_stat_and_new_and_deleted_ones_still_count() {
        let dir = tempfile::tempdir().expect("tempdir");
        let jpeg = |i: u8| vec![0xff, 0xd8, i, 0xff, 0xd9];
        write(dir.path(), 3, "3-1-1.cnk", ChunkFormat::Passthrough, &[(1_000_000, jpeg(0))]);
        write(dir.path(), 3, "3-2-2.cnk", ChunkFormat::Passthrough, &[(2_000_000, jpeg(1))]);
        let src = ChunkSource::new(dir.path());
        let listed = |src: &ChunkSource| {
            src.summaries.lock().expect("lock").last = None; // past LISTING_REUSE
            src.streams().expect("streams").iter().map(|s| (s.id, s.chunks, s.frames, s.end_us)).collect::<Vec<_>>()
        };
        assert_eq!(listed(&src), vec![(3, 2, 2, 2_001)]);
        fs::remove_file(cnk(dir.path(), "3-1-1.cnk")).expect("delete");
        write(dir.path(), 3, "3-3-3.cnk", ChunkFormat::Passthrough, &[(3_000_000, jpeg(2))]);
        assert_eq!(listed(&src), vec![(3, 2, 2, 3_001)]);
    }

    /// A listing holds the cache while it runs (the first one, minutes over a large directory). Asking whether a full
    /// listing has been made never waits on it, and a listing that waited answers from the one it waited for.
    #[test]
    fn a_running_listing_holds_up_neither_the_ready_check_nor_a_second_listing_s_answer() {
        let dir = tempfile::tempdir().expect("tempdir");
        write(dir.path(), 3, "3-1-1.cnk", ChunkFormat::Passthrough, &[(1_000_000, vec![0xff, 0xd8, 0, 0xff, 0xd9])]);
        let delay = std::time::Duration::from_millis(600);
        let src = ChunkSource::slow_listing(dir.path(), delay);
        assert!(!src.summaries_ready());
        let first = {
            let src = src.clone();
            std::thread::spawn(move || src.streams().expect("streams"))
        };
        std::thread::sleep(delay / 6); // the first listing has the cache now
        let asked = std::time::Instant::now();
        assert!(!src.summaries_ready());
        assert!(asked.elapsed() < delay / 6, "the ready check didn't wait for the listing");
        let second = src.streams().expect("streams"); // waits for the first, then answers from it
        assert!(asked.elapsed() < delay * 3 / 2, "it didn't list again");
        assert_eq!(second, first.join().expect("first"));
        assert!(src.summaries_ready());
    }

    #[test]
    fn streams_and_ranges_come_from_the_chunk_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let jpeg = |i: u8| vec![0xff, 0xd8, i, 0xff, 0xd9];
        write(dir.path(), 3, "3-1-1.cnk", ChunkFormat::Passthrough, &[(1_000_000, jpeg(0)), (2_000_000, jpeg(1))]);
        write(
            dir.path(),
            3,
            "3-2-2.cnk",
            ChunkFormat::Passthrough,
            &[(3_000_000, jpeg(2)), (4_000_000, b"garbage".to_vec())],
        );
        write(dir.path(), 9, "9-1-3.cnk", ChunkFormat::Lz4, &[(5_000_000, vec![1, 2, 3])]);
        fs::write(cnk(dir.path(), "x-1-4.cnk"), b"not a chunk").expect("write");
        fs::write(cnk(dir.path(), "7-1-5.cnk"), b"not a chunk").expect("write");
        let src = ChunkSource::new(dir.path());
        let streams = src.streams().expect("streams");
        assert_eq!(
            streams.iter().map(|s| (s.id, s.chunks, s.frames, s.start_us, s.end_us)).collect::<Vec<_>>(),
            vec![(3, 2, 4, 1_000, 4_001), (9, 1, 1, 5_000, 5_001)]
        );
        assert_eq!(streams[0].format, "passthrough");

        let all = src.frames(3, 0, u64::MAX, 100, usize::MAX).expect("frames");
        assert_eq!(all.frames.len(), 3);
        assert_eq!(all.skipped, 1, "garbage is neither JPEG nor Annex B");
        assert_eq!(all.next_us, None);
        assert_eq!(all.frames.iter().map(|f| f.header.capture_us).collect::<Vec<_>>(), vec![1_000, 2_000, 3_000]);
        assert!(
            all.frames
                .iter()
                .all(|f| f.header.payload_type == PayloadType::Jpeg && f.header.keyframe && f.codec == "jpeg")
        );
        assert_eq!(all.frames[1].bytes, jpeg(1));

        let some = src.frames(3, 2_000_000, 4_000_000, 1, usize::MAX).expect("frames");
        assert_eq!(some.frames.len(), 1);
        assert_eq!(some.next_us, Some(3_000), "the cap says where to continue");

        let raw = src.frames(9, 0, u64::MAX, 10, usize::MAX).expect("frames");
        assert_eq!((raw.frames.len(), raw.skipped), (0, 1));
        assert!(ChunkSource::new(&dir.path().join("missing")).streams().expect("empty").is_empty());
    }

    #[test]
    fn an_opus_record_is_skipped() {
        let dir = tempfile::tempdir().expect("tempdir");
        let opus = gop::encode(&[gop::Unit { start_ns: 10_000_000, end_ns: 10_020_000, key: false, bytes: vec![1] }])
            .expect("encode");
        write(dir.path(), 6, "6-1-1.cnk", ChunkFormat::Opus, &[(10_000_000, opus)]);
        let audio = ChunkSource::new(dir.path()).frames(6, 0, 11_000_000, 100, 1 << 20).expect("frames");
        assert_eq!((audio.frames.len(), audio.skipped), (0, 1), "Opus has no player path yet");
    }

    #[test]
    fn annexb_units_are_sniffed_for_codec_and_keyframes() {
        // H.264: SPS (0x67), PPS (0x68), IDR (0x65); then a non-IDR slice (0x41) alone; then an IDR alone.
        let key = [0, 0, 0, 1, 0x67, 1, 2, 0, 0, 0, 1, 0x68, 3, 0, 0, 1, 0x65, 9, 9];
        assert_eq!(annexb_codec(&key, None), Some(("avc", true)));
        assert_eq!(annexb_codec(&[0, 0, 1, 0x41, 5, 5], Some("avc")), Some(("avc", false)));
        assert_eq!(annexb_codec(&[0, 0, 1, 0x65, 5, 5], Some("avc")), Some(("avc", true)), "an IDR alone");
        assert_eq!(annexb_codec(&[0, 0, 1, 0x41, 5, 5], None), None, "no parameter sets and no known codec");
        // An H.264 slice whose second byte happens to be 0x01 is not mistaken for an HEVC parameter set.
        assert_eq!(annexb_codec(&[0, 0, 1, 0x41, 0x01, 5], Some("avc")), Some(("avc", false)));
        // HEVC: VPS (type 32: 0x40 0x01), SPS (33: 0x42 0x01), PPS (34: 0x44 0x01), IDR_W_RADL (19: 0x26 0x01).
        let hkey =
            [0, 0, 0, 1, 0x40, 0x01, 1, 0, 0, 0, 1, 0x42, 0x01, 2, 0, 0, 1, 0x44, 0x01, 3, 0, 0, 1, 0x26, 0x01, 7];
        assert_eq!(annexb_codec(&hkey, None), Some(("hevc", true)));
        // An HEVC TRAIL_R slice (type 1: 0x02 0x01) keeps the stream's codec and is not a keyframe; a CRA (21) is.
        assert_eq!(annexb_codec(&[0, 0, 1, 0x02, 0x01, 8], Some("hevc")), Some(("hevc", false)));
        assert_eq!(annexb_codec(&[0, 0, 1, 0x2a, 0x01, 8], Some("hevc")), Some(("hevc", true)));
        assert_eq!(annexb_codec(b"\xff\xd8", None), None);
        assert_eq!(classify(ChunkFormat::Passthrough, &key, None), Some((PayloadType::Video, true, "avc")));
        assert_eq!(classify(ChunkFormat::ArrowIpc, b"ARROW1", None), Some((PayloadType::Arrow, false, "arrow")));
        assert_eq!(classify(ChunkFormat::Flac, b"fLaC", None), None);
        // Unit records aren't classified: `record_frames` unpacks them first.
        assert_eq!(classify(ChunkFormat::H264Au, b"\0\0\0\x01", None), None);
        assert_eq!(classify(ChunkFormat::Opus, b"OpusHead", None), None);
        assert_eq!((format_name(ChunkFormat::H264Au), format_name(ChunkFormat::Opus)), ("h264_au", "opus"));
        assert_eq!(nal_units(&key).len(), 3);
        assert_eq!(
            nal_units(&key)[0],
            &[0x67, 1, 2],
            "trailing zero bytes before a start code are not part of the unit"
        );
    }

    #[test]
    fn a_range_makes_progress_and_survives_a_vanished_chunk_and_a_corrupt_record() {
        use std::io::{Seek, Write};
        let dir = tempfile::tempdir().expect("tempdir");
        let jpeg = |i: u8| vec![0xff, 0xd8, i, 0xff, 0xd9];
        // Three records in one microsecond, then one more: a cut of 1 still sends all three and continues past them.
        write(
            dir.path(),
            5,
            "5-1-1.cnk",
            ChunkFormat::Passthrough,
            &[(1_000_100, jpeg(0)), (1_000_200, jpeg(1)), (1_000_300, jpeg(2)), (1_001_000, jpeg(3))],
        );
        let src = ChunkSource::new(dir.path());
        let first = src.frames(5, 0, u64::MAX, 1, usize::MAX).expect("frames");
        assert_eq!((first.frames.len(), first.next_us), (3, Some(1_001)));
        let rest = src.frames(5, 1_001_000, u64::MAX, 1, usize::MAX).expect("frames");
        assert_eq!((rest.frames.len(), rest.next_us), (1, None));
        let capped = src.frames(5, 0, u64::MAX, 100, 1).expect("the byte cap cuts too");
        assert_eq!((capped.frames.len(), capped.next_us), (3, Some(1_001)));
        // A record with a flipped payload byte is skipped, not fatal.
        let recs = ChunkReader::open(&cnk(dir.path(), "5-1-1.cnk")).expect("open").records().to_vec();
        let mut f = fs::OpenOptions::new().write(true).open(cnk(dir.path(), "5-1-1.cnk")).expect("open");
        f.seek(std::io::SeekFrom::Start(recs[3].offset + 1)).expect("seek");
        f.write_all(&[0x00]).expect("write");
        drop(f);
        let read = src.frames(5, 0, u64::MAX, 100, usize::MAX).expect("frames");
        assert_eq!((read.frames.len(), read.skipped), (3, 1));
        // A chunk the listing saw but which is gone by the open (a purge) is skipped, for ranges and the list.
        std::os::unix::fs::symlink(dir.path().join("missing.cnk"), cnk(dir.path(), "5-9-9.cnk")).expect("symlink");
        let read = src.frames(5, 0, u64::MAX, 100, usize::MAX).expect("a vanished chunk is not an error");
        assert_eq!(read.frames.len(), 3);
        assert_eq!(src.streams().expect("streams")[0].chunks, 1);
    }
}
