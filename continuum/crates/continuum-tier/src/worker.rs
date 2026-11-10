//! The tier worker's loop (docs/CONTINUUM-STORAGE.md): payloads in, chunks out. It knows nothing about
//! sockets or shared memory: the program feeds it each payload it read from Tier 0 and a clock tick, and it hands
//! back, through [`Outputs`], the ids whose records are now durable (mark those slots `flushed`) and each closed
//! chunk (commit it to the catalog through the broker). So every rule is tested against the in-memory catalog.
//!
//! The rules:
//! - A payload is appended to its stream's open chunk at once, in the chunk's format (LZ4 for raw frames, as is for
//!   MJPEG, FLAC, Arrow IPC, and Opus, which arrive already encoded).
//! - On each tick the flusher syncs what is due ([`FlushSettings::sync_ns`]), cuts a part from each open chunk that
//!   is due one ([`FlushSettings::part_ns`]), and closes chunks whose time is up. Durable ids are reported at once; a
//!   part or a closed chunk is committed, then its remaining ids are reported.
//! - A commit that fails for now (the broker busy or gone) is kept and tried again, at most once every
//!   [`RETRY_NS`], in order, so a broker hiccup never leaves a finished chunk the catalog doesn't know about. A
//!   commit the broker refuses for good can never succeed. A whole chunk's file is set aside (renamed `.rejected`)
//!   and the queue moves on. A later part's chunk is in the catalog already (a purge hid it, say), so its file
//!   stays: the chunk is closed, the records the catalog doesn't hold are moved into a new chunk file
//!   ([`crate::chunk::split_off`]), and that is committed as a chunk of its own. Records that start before
//!   their chunk's last committed frame (a camera's times stepped back) are never sent as a part: the flusher
//!   closes the chunk at the next keyframe and they move to a chunk of their own the same way.
//! - Chunks a crash left open come back from [`crate::recover`], with each stream's newest finished one. Each is
//!   committed once its stream is declared again (its header needs the stream's shape, which the first payload
//!   brings): whole when the catalog doesn't know it, or only the records past what the catalog holds of it
//!   ([`Outputs::committed_frames`]). A payload already in one is reported back rather than appended twice. When
//!   the catalog lacks part of a stream's newest finished chunk, commits were failing as the last run ended, so the
//!   stream's older finished chunks are rechecked the same way, newest first, back to one the catalog holds whole.
//!   A recovered chunk a purge took from the catalog meanwhile isn't committed whole (the catalog would refuse it
//!   and the file would be set aside): the records the catalog still holds as payloads move to a new chunk
//!   ([`Outputs::first_unabsorbed`]).
//! - On shutdown every open chunk is closed and committed.
//! - A raw camera's frames (UYVY, YUYV, or NV12) are encoded as they land, with the `ffmpeg` feature: low-delay
//!   H.264 with a keyframe every second ([`crate::raw::RawEncoder`], hardware first), one access unit per record in a
//!   [`ChunkFormat::H264Au`] chunk, so Tier 1 holds about a gigabyte an hour instead of LZ4's hundred. An access unit
//!   is matched back to its frame by its start time (an encoder can hold the first few frames). A chunk starts on a
//!   keyframe: the flusher closes one only before a keyframe, or once the stream has been quiet, and whenever the
//!   worker itself closes chunks (before a purge, at shutdown, a quiet stream on time) it drains the encoder first and
//!   opens a new one for the next frame. Where no encoder opens, the stream's frames go to Tier 1 as LZ4, logged once.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;

use continuum_core::{StreamId, Uuid};

use crate::TierError;
use crate::chunk::ChunkFormat;
use crate::codecs;
use crate::flush::{Closed, FlushSettings, Flusher, Incoming, StreamShape, new_chunk};
use crate::pcm::{PcmCodec, PcmLayout};
use crate::recover::Recovered;

/// The least time between two tries of a failed commit.
pub const RETRY_NS: u64 = 1_000_000_000;

/// Why a commit didn't happen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommitFailure {
    /// For now (the broker busy, the connection down): try again.
    Retry(String),
    /// For good (the broker refused the chunk itself): trying again can't help.
    Reject(String),
}

/// What became of a payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Appended {
    /// Appended to its stream's open chunk.
    Appended,
    /// Already in a chunk recovered at startup: not appended again.
    Recovered,
    /// Preserved durably in a layout-unknown quarantine, without a guessed catalog entry.
    Quarantined,
    /// A native H.264 frame after its stream's chunk closed away from a keyframe (a quiet stream, a purge, a
    /// restart), before the next keyframe: nothing in Tier 1 could decode it, so it's dropped and the stream's next
    /// chunk starts on its next keyframe. The broker records it as a gap.
    Undecodable,
}

/// One payload as the program read it from Tier 0.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Payload {
    /// Its id.
    pub id: Uuid,
    /// Its stream.
    pub stream: StreamId,
    /// Its start, UTC nanoseconds.
    pub start_ns: u64,
    /// Its end; equal to `start_ns` for an instant.
    pub end_ns: u64,
    /// Its sequence number.
    pub seq: u32,
    /// Its bytes as the producer wrote them.
    pub bytes: Vec<u8>,
    /// Source audio rate from this payload, or zero for a legacy unknown-rate producer.
    pub pcm_rate_hz: u32,
    /// What capture attested about it (device sequence, capture epoch, stamp), or `None` for a producer that didn't
    /// say (a capture epoch of zero). Kept with its chunk after the header is absorbed.
    pub provenance: Option<continuum_catalog::FrameProvenance>,
}

/// Per-frame camera capture information. Size/stride belong to this frame, not to its long-lived stream id.
/// A non-increasing device sequence on a distinct payload marks a new capture epoch, including rate-only restarts.
/// There is no per-frame nominal video rate in the wire schema; timestamps are never used to guess one here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CameraFrameInfo {
    /// Captured width in pixels.
    pub width: u32,
    /// Captured height in pixels.
    pub height: u32,
    /// Bytes per row for raw video (zero means packed).
    pub stride: u32,
    /// Whether consumers turn the native pixels upright by rotating them 180 degrees.
    pub flip: bool,
    /// The camera worker's device sequence, which can restart at zero or wrap.
    pub device_seq: u64,
}

#[derive(Clone, Debug)]
struct VideoState {
    source: StreamShape,
    size: (u32, u32, u32, bool),
    device_seq: Option<u64>,
    id: Uuid,
}

/// Where the loop's results go: the broker, in the program; a catalog and a set, in tests.
pub trait Outputs {
    /// These payloads' records are durable on disk: mark their slots `flushed`.
    ///
    /// # Errors
    ///
    /// Why they couldn't be marked (the arena went away). The loop logs it; the slots stay unflushed, and the
    /// broker reclaims them only when it must.
    fn flushed(&mut self, ids: &[Uuid]) -> Result<(), String>;

    /// Commits a chunk, or a part of one ([`Closed::first_frame`]), to the catalog, which compacts the absorbed
    /// frames into it.
    ///
    /// # Errors
    ///
    /// Why it couldn't be committed: [`CommitFailure::Retry`] and the loop tries again; [`CommitFailure::Reject`]
    /// and it sets a whole chunk aside and calls [`Outputs::rejected`], or moves a part's records to a new chunk.
    fn commit(&mut self, closed: &Closed) -> Result<(), CommitFailure>;

    /// A chunk the broker refused for good was set aside (its file renamed to `aside`): forget its payloads.
    fn rejected(&mut self, _closed: &Closed, _aside: &Path) {}

    /// How many frames the catalog holds of `chunk`: `None` when it doesn't know the chunk. Asked once for each
    /// chunk recovered at startup.
    ///
    /// # Errors
    ///
    /// Why the catalog couldn't be asked. The stream then takes no payloads this run, so nothing is committed twice.
    fn committed_frames(&mut self, _chunk: Uuid) -> Result<Option<usize>, String> {
        Ok(None)
    }

    /// The index of the first of `ids` the catalog still holds as a payload of its own, waiting to be absorbed into
    /// a chunk: `None` when it holds none of them that way. Asked about the records of a recovered chunk the catalog
    /// doesn't know, to tell a chunk it never saw (the first record, index 0) from one a purge took: there the
    /// records the chunk had absorbed are gone, and only the ones that arrived after its last part are left.
    ///
    /// # Errors
    ///
    /// Why the catalog couldn't be asked. The stream then takes no payloads this run.
    fn first_unabsorbed(&mut self, _ids: &[Uuid]) -> Result<Option<usize>, String> {
        Ok(Some(0))
    }

    /// These payloads are in a recovered chunk's frames the catalog holds already: they are safe.
    fn settled(&mut self, _ids: &[Uuid]) {}
}

/// What the loop has done, for the program's health line.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// Payloads appended.
    pub appended: u64,
    /// Commits made: chunks, and parts of them.
    pub committed: u64,
    /// Times a refused part's records were moved into a new chunk.
    pub rehomed: u64,
    /// Payloads reported durable.
    pub flushed: u64,
    /// Commits that failed and were queued to try again.
    pub commit_retries: u64,
    /// Chunks the broker refused for good, set aside.
    pub rejected: u64,
    /// Chunks recovered at startup and committed.
    pub recovered: u64,
    /// Older finished chunks opened at startup because the catalog lacked part of a newer one.
    pub walked_back: u64,
    /// Payloads refused (an undeclared stream, a codec error).
    pub refused: u64,
    /// Microphone batches stored as LZ4 because their samples weren't exact integers ([`crate::pcm`]).
    pub pcm_fallbacks: u64,
    /// Native H.264 frames dropped while their stream waited for a keyframe to start a chunk on.
    pub undecodable: u64,
    /// Microphone chunks committed as PCM because they couldn't become Opus as they closed (an unknown rate, a
    /// layout Opus can't keep, or a codec error).
    pub opus_fallbacks: u64,
}

/// What rechecking one chunk file found at startup against the catalog came to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rechecked {
    /// The catalog holds all of it: nothing to commit.
    Whole,
    /// The catalog lacked part of it, which was committed or queued to be.
    Committed,
    /// It couldn't be rechecked or committed, or the catalog refused it.
    Stopped,
}

/// The tier worker's loop over one chunk directory.
pub struct WorkerLoop<O: Outputs> {
    /// The chunk directory.
    dir: std::path::PathBuf,
    flusher: Flusher,
    out: O,
    formats: BTreeMap<StreamId, (ChunkFormat, Option<PcmLayout>)>,
    /// Original source shapes, before a raw stream is upgraded to H.264 for storage.
    sources: BTreeMap<StreamId, StreamShape>,
    video: BTreeMap<StreamId, VideoState>,
    /// Microphone streams whose fallback to LZ4 has been logged (once each; `Counts::pcm_fallbacks` counts them all).
    pcm_warned: BTreeSet<StreamId>,
    /// Microphone streams whose first chunk kept as PCM has been logged (`Counts::opus_fallbacks` counts them all).
    opus_warned: BTreeSet<StreamId>,
    /// Chunks and parts whose commit failed, oldest first.
    uncommitted: VecDeque<Closed>,
    /// The file of a chunk whose part was refused, while its remaining records are gathered: parts of it that
    /// turn up meanwhile go to `gathered` instead of the catalog.
    refused: Option<std::path::PathBuf>,
    gathered: Vec<Closed>,
    /// When a queued commit was last tried.
    last_retry_ns: u64,
    /// Chunks recovered at startup, by stream, waiting for the stream's shape.
    recovered: BTreeMap<StreamId, Vec<Recovered>>,
    /// Payloads in a recovered chunk: not appended again.
    recovered_ids: BTreeSet<Uuid>,
    quarantined: BTreeSet<Uuid>,
    recovery_errors: BTreeMap<StreamId, String>,
    counts: Counts,
    /// Each raw camera stream being encoded.
    #[cfg(feature = "ffmpeg")]
    raw: BTreeMap<StreamId, RawStream>,
}

/// The frame rate a raw camera's encoder is opened for, and so its GOP: a keyframe every this many frames (a second at
/// the 30 fps cameras send; the access units keep each frame's own times whatever the real rate).
#[cfg(feature = "ffmpeg")]
pub const RAW_FPS: u32 = 30;

/// A raw camera stream's encoder, and the frames given it whose access units haven't come back yet.
#[cfg(feature = "ffmpeg")]
struct RawStream {
    layout: crate::raw::RawLayout,
    /// None between a close the worker made and the stream's next frame, which opens a new one (on a keyframe).
    encoder: Option<crate::raw::RawEncoder>,
    /// (id, start, end, seq) of each frame encoded and not yet appended, in order.
    waiting: VecDeque<(Uuid, u64, u64, u32)>,
    /// The next access unit's record seq. The H.264 this worker encodes decodes in its own output order, so its
    /// records count that order, consecutively; a camera's frame seq is 0 for a raw publish, and its jumps (driver
    /// drops) don't break this stream. Every record's seq was 0 on the Mac, so the gateway took each P-frame for a
    /// gap and Review showed only keyframes.
    next_seq: u32,
}

impl<O: Outputs> WorkerLoop<O> {
    /// A loop writing chunks under `dir`.
    ///
    /// # Errors
    ///
    /// An I/O error creating the directory.
    pub fn new(dir: &Path, settings: FlushSettings, out: O) -> Result<Self, TierError> {
        let flusher = Flusher::new(dir, settings)?;
        let mut quarantined = BTreeSet::new();
        let recovery_errors = BTreeMap::new();
        for path in crate::layout::chunk_paths(dir, None)? {
            if path.extension().is_some_and(|ext| ext == "layout-unknown") {
                let reader = crate::ChunkReader::open(&path)?;
                std::fs::File::open(&path)?.sync_all()?;
                std::fs::File::open(path.parent().unwrap_or(dir))?.sync_all()?;
                quarantined.extend(reader.records().iter().map(|record| record.id));
            }
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            flusher,
            out,
            formats: BTreeMap::new(),
            sources: BTreeMap::new(),
            video: BTreeMap::new(),
            pcm_warned: BTreeSet::new(),
            opus_warned: BTreeSet::new(),
            uncommitted: VecDeque::new(),
            refused: None,
            gathered: Vec::new(),
            last_retry_ns: 0,
            recovered: BTreeMap::new(),
            recovered_ids: BTreeSet::new(),
            quarantined,
            recovery_errors,
            counts: Counts::default(),
            #[cfg(feature = "ffmpeg")]
            raw: BTreeMap::new(),
        })
    }

    /// Takes the chunks [`crate::recover`] found, to commit as their streams are declared.
    pub fn recovered(&mut self, chunks: Vec<Recovered>) {
        for r in chunks {
            self.recovered_ids.extend(r.records.iter().map(|rec| rec.id));
            self.recovered.entry(r.stream).or_default().push(r);
        }
    }

    /// Whether a payload is in a recovered chunk.
    #[must_use]
    pub fn is_recovered(&self, id: Uuid) -> bool {
        self.recovered_ids.contains(&id)
    }

    fn layout_error(path: &Path) -> String {
        format!(
            "raw camera chunk {} has no durable pixel format/size/stride; preserved without catalog commit; restore its original capture layout and recover it manually; later frames continue",
            path.display()
        )
    }

    /// Whether this payload is already preserved in a durable layout-unknown quarantine.
    #[must_use]
    pub fn is_quarantined(&self, id: Uuid) -> bool {
        self.quarantined.contains(&id)
    }

    /// A failed quarantine durability operation prevents this stream from acknowledging unsafe bytes.
    #[must_use]
    pub fn recovery_error(&self, stream: StreamId) -> Option<&str> {
        self.recovery_errors.get(&stream).map(String::as_str)
    }

    /// Starts following a stream, and commits the chunks recovered for it.
    pub fn declare(&mut self, stream: StreamId, shape: StreamShape) {
        let camera = shape.raw.is_some()
            || shape.format == continuum_core::PayloadFormat::EncodedVideo
            || shape.shape_name == "MJPEG";
        self.sources.insert(stream, shape.clone());
        #[cfg(feature = "ffmpeg")]
        let shape = self.raw_shape(stream, shape);
        self.formats.insert(stream, (shape.chunk_format, shape.pcm));
        let found = self.recovered.remove(&stream).unwrap_or_default();
        // The oldest of them is the stream's newest finished chunk from before this run (or, with none, the first it
        // left open). When the catalog lacks part of it, commits were failing as the last run ended.
        let oldest = found.first().map(|r| r.path.clone());
        let mut walk_back = false;
        for (i, r) in found.into_iter().enumerate() {
            let lacking = self.commit_recovered_file(stream, &shape, camera, r) == Rechecked::Committed;
            walk_back |= i == 0 && lacking;
        }
        if let (true, Some(oldest)) = (walk_back, oldest) {
            self.walk_back(stream, &shape, camera, &oldest);
        }
        self.flusher.declare(stream, shape);
    }

    /// Parts queued behind a failing commit can belong to more than one chunk file, and a start hands back only each
    /// stream's newest finished one. So when the catalog lacked part of that one (`newest`), this goes back through
    /// the stream's older finished chunks, newest first, committing what the catalog lacks of each. It stops at the
    /// first one the catalog holds whole, so a start after a clean stop opens no older file, and at the first one the
    /// catalog refuses or can't be asked about, so chunk files a catalog no longer knows (after a reset, say) aren't
    /// each opened and set aside.
    fn walk_back(&mut self, stream: StreamId, shape: &StreamShape, camera: bool, newest: &Path) {
        let older = match crate::recover::older_chunks(&self.dir, stream, newest) {
            Ok(paths) => paths,
            Err(e) => {
                tracing::warn!("listing stream {}'s older chunks failed: {e}", stream.0);
                return;
            }
        };
        for path in older {
            let r = match crate::recover::finished(&path) {
                Ok(Some(r)) if r.stream == stream => r,
                Ok(_) => continue,
                Err(e) => {
                    tracing::warn!("reading older chunk {} failed: {e}", path.display());
                    return;
                }
            };
            self.counts.walked_back += 1;
            self.recovered_ids.extend(r.records.iter().map(|rec| rec.id));
            if self.commit_recovered_file(stream, shape, camera, r) != Rechecked::Committed {
                return;
            }
        }
    }

    /// Commits what the catalog lacks of one chunk file found at startup, and says which it was.
    fn commit_recovered_file(
        &mut self,
        stream: StreamId,
        shape: &StreamShape,
        camera: bool,
        r: Recovered,
    ) -> Rechecked {
        // What the catalog holds of it already: parts committed before the crash.
        let asked = match crate::layout::chunk_id_of(&r.path) {
            Some(id) => self.out.committed_frames(id),
            None => Ok(None),
        };
        // A chunk the catalog doesn't know is one it never saw, or one a purge took while the last run was down
        // (a hidden chunk and an unknown one answer alike). Its records the catalog still holds as payloads say
        // which: all of them for a new chunk, and only those past its last part for one a purge took.
        let asked = asked.and_then(|held| match held {
            Some(held) => Ok((held, None)),
            None => {
                let ids: Vec<Uuid> = r.records.iter().map(|rec| rec.id).collect();
                self.out.first_unabsorbed(&ids).map(|first| (0, first.filter(|first| *first > 0)))
            }
        });
        let (held, purged_before) = match asked {
            Ok(found) => found,
            Err(e) => {
                let why = format!("asking the catalog about recovered chunk {} failed: {e}", r.path.display());
                tracing::warn!("{why}");
                self.recovery_errors.insert(stream, why);
                return Rechecked::Stopped;
            }
        };
        if let Some(keep) = purged_before {
            self.rehome_recovered(stream, &r, keep);
            return Rechecked::Stopped;
        }
        let from = held.min(r.records.len());
        let known: Vec<Uuid> = r.records[..from].iter().map(|rec| rec.id).collect();
        self.out.settled(&known);
        if from == r.records.len() {
            return Rechecked::Whole;
        }
        if camera && r.format == ChunkFormat::Lz4 && from == 0 {
            let aside = r.path.with_extension(format!("{}.layout-unknown", uuid::Uuid::now_v7()));
            // A hard link refuses to overwrite an existing quarantine; only remove the original after success.
            let renamed = (|| -> std::io::Result<()> {
                std::fs::File::open(&r.path)?.sync_all()?;
                std::fs::hard_link(&r.path, &aside)?;
                std::fs::File::open(aside.parent().expect("chunk directory"))?.sync_all()?;
                std::fs::remove_file(&r.path)?;
                std::fs::File::open(aside.parent().expect("chunk directory"))?.sync_all()
            })();
            let mut why = Self::layout_error(if renamed.is_ok() { &aside } else { &r.path });
            if let Err(error) = renamed {
                why.push_str(&format!("; quarantine durability failed: {error}"));
                self.recovery_errors.insert(stream, why.clone());
            } else {
                self.quarantined.extend(r.records.iter().map(|record| record.id));
            }
            tracing::warn!("{why}");
            for record in r.records {
                self.recovered_ids.remove(&record.id);
            }
            return Rechecked::Stopped;
        }
        let records = r.records[from..].to_vec();
        let absorbed: Vec<Uuid> = crate::flush::catalog_records(&records).iter().map(|rec| rec.id).collect();
        match crate::flush::new_part(&r.path, stream, shape, &r.records, from, r.created_ns) {
            Ok(chunk) => {
                // Frames the chunk file has no provenance record for stay unknown.
                let provenance = crate::flush::provenance_of(&absorbed, &r.provenance);
                let closed = Closed {
                    path: r.path,
                    stream,
                    records,
                    chunk,
                    first_frame: u32::try_from(from).unwrap_or(u32::MAX),
                    more: false,
                    absorbed,
                    durable: Vec::new(),
                    provenance,
                    stepped_back: false,
                };
                self.counts.recovered += 1;
                let refused = (self.counts.rejected, self.counts.rehomed);
                self.finish(closed);
                if refused == (self.counts.rejected, self.counts.rehomed) {
                    Rechecked::Committed
                } else {
                    Rechecked::Stopped
                }
            }
            Err(e) => {
                tracing::warn!("recovered chunk {} has no header: {e}", r.path.display());
                Rechecked::Stopped
            }
        }
    }

    /// Appends one payload, committing a chunk it closed on the way.
    ///
    /// # Errors
    ///
    /// [`TierError::UnknownStream`] for an undeclared stream, a codec error, or a chunk or I/O error; the payload
    /// is counted as refused and the loop carries on.
    pub fn payload(&mut self, p: &Payload, now_ns: u64) -> Result<Appended, TierError> {
        self.accept(p, None, now_ns)
    }

    /// Appends a camera frame using its current capture layout, including across a capture worker restart.
    /// The source shape is the incoming frame's shape, before any Tier 1 raw-video encoding.
    ///
    /// # Errors
    ///
    /// As [`Self::payload`]. A boundary first drains and commits the old encoder/chunk before replacing its layout.
    pub fn video_payload(
        &mut self,
        p: &Payload,
        source: StreamShape,
        frame: CameraFrameInfo,
        now_ns: u64,
    ) -> Result<Appended, TierError> {
        self.accept(p, Some((source, frame)), now_ns)
    }

    fn accept(
        &mut self,
        p: &Payload,
        camera: Option<(StreamShape, CameraFrameInfo)>,
        now_ns: u64,
    ) -> Result<Appended, TierError> {
        if let Some(why) = self.recovery_error(p.stream) {
            return Err(TierError::Codec(why.to_owned()));
        }
        if self.is_quarantined(p.id) {
            return Ok(Appended::Quarantined);
        }
        if self.recovered_ids.contains(&p.id) {
            return Ok(Appended::Recovered);
        }
        let result = self.prepare_video(p, camera, now_ns).and_then(|()| {
            if let Some(provenance) = p.provenance {
                self.flusher.attest(p.stream, p.id, provenance);
            }
            self.append(p, now_ns)
        });
        if result.is_err() {
            self.counts.refused += 1;
            // A chunk this append closed before the next one failed to open is parked in the flusher: commit it.
            for c in self.flusher.parked() {
                self.finish(c);
            }
        }
        result.map(|appended| if appended { Appended::Appended } else { Appended::Undecodable })
    }

    fn prepare_video(
        &mut self,
        p: &Payload,
        camera: Option<(StreamShape, CameraFrameInfo)>,
        now_ns: u64,
    ) -> Result<(), TierError> {
        #[cfg(not(feature = "ffmpeg"))]
        let _ = now_ns;
        let (source, mut size, device_seq) = match camera {
            Some((source, frame)) => {
                (source, (frame.width, frame.height, frame.stride, frame.flip), Some(frame.device_seq))
            }
            None => {
                let Some(source) = self.sources.get(&p.stream).cloned() else { return Ok(()) };
                let size = source.raw.map(|raw| (raw.width, raw.height, raw.stride, false));
                if source.shape_name != "MJPEG" && size.is_none() {
                    return Ok(());
                }
                (source, size.unwrap_or_default(), None)
            }
        };
        // JPEG is self-describing. Read only its headers, so legacy callers without CaptureInfo still split sizes.
        if source.shape_name == "MJPEG" {
            let mut jpeg = zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(&p.bytes));
            if jpeg.decode_headers().is_ok()
                && let Some(info) = jpeg.info()
            {
                size = (u32::from(info.width), u32::from(info.height), 0, size.3);
            }
        }
        if size.0 == 0 || size.1 == 0 {
            return Ok(());
        }
        let boundary = self.sources.get(&p.stream).is_some_and(|old| old != &source)
            || self.video.get(&p.stream).is_some_and(|old| {
                old.source != source
                    || old.size != size
                    || (old.id != p.id && device_seq.zip(old.device_seq).is_some_and(|(new, old)| new <= old))
            });
        if boundary {
            #[cfg(feature = "ffmpeg")]
            self.drain(p.stream, now_ns)?;
            if let Some(closed) = self.flusher.close(p.stream)? {
                self.finish(closed);
            }
            #[cfg(feature = "ffmpeg")]
            self.raw.remove(&p.stream);
            self.declare(p.stream, source.clone());
        } else if !self.sources.contains_key(&p.stream) {
            self.declare(p.stream, source.clone());
        }
        self.flusher.camera_layout(
            p.stream,
            CameraFrameInfo {
                width: size.0,
                height: size.1,
                stride: size.2,
                flip: size.3,
                device_seq: device_seq.unwrap_or(0),
            },
        );
        self.video.insert(p.stream, VideoState { source, size, device_seq, id: p.id });
        Ok(())
    }

    /// Whether it was appended: a native H.264 P-frame with no chunk open to start from is dropped instead.
    fn append(&mut self, p: &Payload, now_ns: u64) -> Result<bool, TierError> {
        let (format, pcm) = *self.formats.get(&p.stream).ok_or(TierError::UnknownStream(p.stream))?;
        let native_h264 = format == ChunkFormat::H264Au
            && self.sources.get(&p.stream).is_some_and(|source| source.shape_name == "H264AnnexB");
        #[cfg(feature = "ffmpeg")]
        if format == ChunkFormat::H264Au && !native_h264 {
            return self.encode(p, now_ns).map(|()| true);
        }
        let key = !native_h264 || crate::gop::h264_idr(&p.bytes);
        if !key && !self.flusher.open_streams().contains(&p.stream) {
            self.counts.undecodable += 1;
            return Ok(false);
        }
        let encoded;
        let bytes: &[u8] = match (format, pcm) {
            (ChunkFormat::H264Au, _) if native_h264 => {
                encoded = crate::gop::encode(&[crate::gop::Unit {
                    start_ns: p.start_ns,
                    end_ns: p.end_ns,
                    key,
                    bytes: p.bytes.clone(),
                }])?;
                &encoded
            }
            (ChunkFormat::Lz4, _) => {
                encoded = codecs::lz4_compress(&p.bytes);
                &encoded
            }
            // A PCM batch stream whose layout isn't known: still a PCM batch record, so it reads back.
            (ChunkFormat::PcmBatch, None) => {
                encoded = crate::pcm::encode_lz4(&p.bytes, PcmLayout { channels: 0, stride: 0 }, 0);
                &encoded
            }
            (ChunkFormat::PcmBatch, Some(layout)) => {
                let (packed, codec) = if p.pcm_rate_hz == 0 {
                    crate::pcm::encode_batch(&p.bytes, layout)?
                } else {
                    crate::pcm::encode_batch_at_rate(
                        &p.bytes,
                        crate::pcm::PcmFormat { sample_rate_hz: p.pcm_rate_hz, layout },
                    )?
                };
                self.pcm_codec(p.stream, codec);
                encoded = packed;
                &encoded
            }
            // MJPEG frames, FLAC, Arrow IPC, and Opus arrive in the chunk's format already, and an H.264 stream's
            // records are made by `encode` (without the `ffmpeg` feature no stream is one). No stream is declared
            // AV1: only the archive pass writes it, over a closed H.264 chunk (crate::av1).
            (
                ChunkFormat::Passthrough
                | ChunkFormat::Flac
                | ChunkFormat::ArrowIpc
                | ChunkFormat::Opus
                | ChunkFormat::H264Au
                | ChunkFormat::Av1Au
                | ChunkFormat::Mp4,
                _,
            ) => &p.bytes,
        };
        // What this payload closes or cuts is committed before the payload is appended. A part the catalog refuses
        // has its chunk closed and its records moved away; a keyframe already appended would move with them, and
        // the frames after it would be dropped for want of one.
        if let Some(c) = self.flusher.before(p.stream, p.start_ns, key, now_ns)? {
            self.finish(c);
        }
        let incoming =
            Incoming { id: p.id, stream: p.stream, start_ns: p.start_ns, end_ns: p.end_ns, seq: p.seq, bytes, key };
        let (_, closed) = self.flusher.append(incoming, now_ns)?;
        self.counts.appended += 1;
        if let Some(c) = closed {
            self.finish(c);
        }
        Ok(true)
    }

    /// Counts a microphone batch that fell back to LZ4, and logs the first for each stream (a stream may alternate,
    /// since a batch of digital silence is exact even under a gain).
    fn pcm_codec(&mut self, stream: StreamId, codec: PcmCodec) {
        if codec != PcmCodec::Lz4 {
            return;
        }
        self.counts.pcm_fallbacks += 1;
        if self.pcm_warned.insert(stream) {
            tracing::warn!(
                "stream {} audio went to Tier 1 as LZ4: its f32 samples aren't exact 16 or 24 bit \
                 integers (a stream volume or resampling in PipeWire?); later fallbacks are only counted",
                stream.0
            );
        }
    }

    /// A raw camera stream's shape: an H.264 stream when an encoder opens for its layout, else as it came (LZ4),
    /// with why, once.
    #[cfg(feature = "ffmpeg")]
    fn raw_shape(&mut self, stream: StreamId, shape: StreamShape) -> StreamShape {
        let Some(layout) = shape.raw else { return shape };
        if self.raw.contains_key(&stream) {
            return StreamShape { chunk_format: ChunkFormat::H264Au, ..shape };
        }
        match crate::raw::RawEncoder::open(layout, RAW_FPS, crate::h264::H264_ENCODERS) {
            Ok(encoder) => {
                tracing::info!(
                    "stream {} ({}x{} {:?}) goes to Tier 1 as H.264 by {}",
                    stream.0,
                    layout.width,
                    layout.height,
                    layout.format,
                    encoder.encoder_name()
                );
                self.raw.insert(
                    stream,
                    RawStream { layout, encoder: Some(encoder), waiting: VecDeque::new(), next_seq: 1 },
                );
                StreamShape { chunk_format: ChunkFormat::H264Au, ..shape }
            }
            Err(e) => {
                tracing::warn!("stream {} raw frames go to Tier 1 as LZ4: no H.264 encoder opens: {e}", stream.0);
                shape
            }
        }
    }

    /// Encodes a raw frame and appends the access units that came back.
    #[cfg(feature = "ffmpeg")]
    fn encode(&mut self, p: &Payload, now_ns: u64) -> Result<(), TierError> {
        let rs = self.raw.get_mut(&p.stream).ok_or(TierError::UnknownStream(p.stream))?;
        if rs.encoder.is_none() {
            rs.encoder = Some(crate::raw::RawEncoder::open(rs.layout, RAW_FPS, crate::h264::H264_ENCODERS)?);
        }
        let encoder = rs.encoder.as_mut().expect("opened above");
        let units = encoder.encode(&p.bytes, p.start_ns, p.end_ns)?;
        rs.waiting.push_back((p.id, p.start_ns, p.end_ns, p.seq));
        self.append_units(p.stream, units, now_ns)
    }

    /// Appends access units, each as its frame's record.
    #[cfg(feature = "ffmpeg")]
    fn append_units(
        &mut self,
        stream: StreamId,
        units: Vec<crate::encode::AccessUnit>,
        now_ns: u64,
    ) -> Result<(), TierError> {
        for au in units {
            let rs = self.raw.get_mut(&stream).ok_or(TierError::UnknownStream(stream))?;
            let (id, start_ns, end_ns, _frame_seq) = match rs.waiting.pop_front() {
                Some(w) if w.1 == au.start_ns => w,
                other => {
                    return Err(TierError::Codec(format!(
                        "stream {}: an access unit at {} doesn't match the next frame waiting ({:?})",
                        stream.0,
                        au.start_ns,
                        other.map(|w| w.1)
                    )));
                }
            };
            let seq = rs.next_seq;
            rs.next_seq = rs.next_seq.wrapping_add(1).max(1);
            let unit = crate::gop::Unit { start_ns, end_ns, key: au.key, bytes: au.bytes };
            let bytes = crate::gop::encode(&[unit])?;
            let incoming = Incoming { id, stream, start_ns, end_ns, seq, bytes: &bytes, key: au.key };
            let (_, closed) = self.flusher.append(incoming, now_ns)?;
            self.counts.appended += 1;
            if let Some(c) = closed {
                self.finish(c);
            }
        }
        Ok(())
    }

    /// Drains a raw stream's encoder (the frames it held go in) and drops it, so the stream's next frame opens a new
    /// one on a keyframe: called before the worker closes the stream's chunk itself.
    #[cfg(feature = "ffmpeg")]
    fn drain(&mut self, stream: StreamId, now_ns: u64) -> Result<(), TierError> {
        let Some(encoder) = self.raw.get_mut(&stream).and_then(|rs| rs.encoder.take()) else { return Ok(()) };
        let units = encoder.finish()?;
        self.append_units(stream, units, now_ns)
    }

    #[cfg(feature = "ffmpeg")]
    fn drain_all(&mut self) -> Result<(), TierError> {
        let now = crate::service::now_utc_ns();
        for stream in self.raw.keys().copied().collect::<Vec<_>>() {
            self.drain(stream, now)?;
        }
        Ok(())
    }

    /// Lets time pass: retries failed commits, syncs what is due, and closes chunks whose time is up.
    ///
    /// # Errors
    ///
    /// An I/O error syncing or closing a chunk.
    pub fn tick(&mut self, now_ns: u64) -> Result<(), TierError> {
        if now_ns.saturating_sub(self.last_retry_ns) >= RETRY_NS {
            self.last_retry_ns = now_ns;
            self.retry();
        }
        let durable = self.flusher.sync_due(now_ns)?;
        self.report(&durable);
        #[cfg(feature = "ffmpeg")]
        for stream in self.flusher.due(now_ns) {
            self.drain(stream, now_ns)?; // a quiet camera: its held frames go in before its chunk closes
        }
        for c in self.flusher.rotate(now_ns)? {
            self.finish(c);
        }
        Ok(())
    }

    /// Closes and commits every open chunk (shutdown). Commits that still fail are left in
    /// [`WorkerLoop::uncommitted`]; their files are finished, so the next start's recovery leaves them alone and the
    /// program must commit them first (see the module doc).
    ///
    /// # Errors
    ///
    /// A chunk or I/O error.
    pub fn shutdown(&mut self) -> Result<(), TierError> {
        #[cfg(feature = "ffmpeg")]
        self.drain_all()?;
        for c in self.flusher.close_all()? {
            self.finish(c);
        }
        self.retry();
        Ok(())
    }

    /// The streams with a chunk open now.
    #[must_use]
    pub fn open_streams(&self) -> Vec<StreamId> {
        self.flusher.open_streams()
    }

    /// Closes and commits every open chunk now, and tries the queued commits (before a purge, so no purged payload
    /// is left in a file that's still being written).
    ///
    /// # Errors
    ///
    /// A chunk or I/O error.
    pub fn close_open(&mut self) -> Result<(), TierError> {
        #[cfg(feature = "ffmpeg")]
        self.drain_all()?;
        for c in self.flusher.close_all()? {
            self.finish(c);
        }
        self.retry();
        Ok(())
    }

    /// Commits a chunk recovered at startup, as if it had just closed.
    pub fn commit_recovered(&mut self, closed: Closed) {
        self.finish(closed);
    }

    /// Chunks closed but not yet committed.
    #[must_use]
    pub fn uncommitted(&self) -> usize {
        self.uncommitted.len()
    }

    /// Payloads appended but not yet durable.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.flusher.pending()
    }

    /// What the loop has done.
    #[must_use]
    pub fn counts(&self) -> Counts {
        self.counts
    }

    /// The outputs, for a test to look at.
    #[must_use]
    pub fn outputs(&self) -> &O {
        &self.out
    }

    /// The outputs, for the program to keep its own state in.
    pub fn outputs_mut(&mut self) -> &mut O {
        &mut self.out
    }

    /// A chunk closed, or a part was cut: report what that made durable (those records are on disk either way), then
    /// commit it, or queue it when the commit fails. Commits stay in order: one waits behind any queued before it.
    fn finish(&mut self, closed: Closed) {
        let closed = self.opus_at_close(closed);
        let durable = closed.durable.clone();
        self.report(&durable);
        if self.refused.as_deref() == Some(closed.path.as_path()) {
            self.gathered.push(closed);
            return;
        }
        if closed.stepped_back && closed.first_frame > 0 {
            self.step_back(&closed);
            return;
        }
        if !self.uncommitted.is_empty() {
            self.uncommitted.push_back(closed);
            self.retry();
            return;
        }
        match self.out.commit(&closed) {
            Ok(()) => self.counts.committed += 1,
            Err(CommitFailure::Retry(e)) => {
                tracing::warn!("committing {} failed, will try again: {e}", closed.path.display());
                self.counts.commit_retries += 1;
                self.uncommitted.push_back(closed);
            }
            Err(CommitFailure::Reject(e)) => self.reject(closed, &e),
        }
    }

    /// A chunk closed with records that start before its last committed frame: the stream's times stepped back
    /// (capture started a network camera's clock fit over). The catalog would refuse them as the chunk's last part,
    /// so they aren't sent: they move to a chunk of their own, which the catalog takes whole, as after a refused
    /// part ([`WorkerLoop::rehome`]). The flusher closed the chunk before the keyframe that follows, so that frame
    /// opens the next chunk and nothing after it is dropped.
    fn step_back(&mut self, closed: &Closed) {
        tracing::info!(
            "stream {}'s frame times stepped back; the {} records since chunk {}'s last part move to a new chunk",
            closed.stream.0,
            closed.records.len(),
            closed.path.display()
        );
        self.counts.rehomed += 1;
        match self.rehome(&closed.path, closed.stream, closed.first_frame as usize) {
            Ok(Some(moved)) => self.finish(moved),
            Ok(None) => self.out.rejected(closed, &closed.path),
            Err(e) => {
                tracing::warn!("moving the records of {} failed: {e}", closed.path.display());
                self.out.rejected(closed, &closed.path);
            }
        }
    }

    /// A microphone's PCM batch chunk becomes Opus before it's committed: the same records, payload ids,
    /// times, and frame order, in the same file ([`crate::opus::pcm_chunk_to_opus`]). One that can't (an unknown
    /// rate, a layout Opus can't keep, a codec error) is committed as PCM; the first for each stream is logged with
    /// why, and the rest are counted in [`Counts::opus_fallbacks`]. Every other chunk passes through.
    fn opus_at_close(&mut self, mut closed: Closed) -> Closed {
        if !self.formats.get(&closed.stream).is_some_and(|(format, _)| *format == ChunkFormat::PcmBatch) {
            return closed;
        }
        let converted = crate::opus::pcm_chunk_to_opus(&closed.path).and_then(|records| {
            let length = std::fs::metadata(&closed.path)?.len();
            Ok((records, length))
        });
        match converted {
            Ok((records, length)) => {
                if let Some(continuum_core::StorageLocator::Chunk { length: l, .. }) =
                    &mut closed.chunk.payload.location
                {
                    *l = length;
                }
                let ordered = crate::flush::catalog_records(&records);
                closed.chunk.frames = ordered.iter().map(|r| r.frame()).collect();
                closed.absorbed = ordered.iter().map(|r| r.id).collect();
                closed.records = records;
            }
            Err(e) => {
                self.counts.opus_fallbacks += 1;
                if self.opus_warned.insert(closed.stream) {
                    tracing::warn!(
                        "stream {} chunk {} stays PCM, not Opus: {e}; later ones are only counted",
                        closed.stream.0,
                        closed.path.display()
                    );
                }
            }
        }
        closed
    }

    /// A commit the broker refused for good. The rest of the chunk is gathered first: its file is closed if it is
    /// still open, and its parts still queued, or cut meanwhile, are taken back. Then a whole chunk (nothing of it
    /// is in the catalog) is set aside: its file is renamed `.rejected` (kept for a person to look at; recovery
    /// skips it), and its payloads are forgotten. A later part's chunk is in the catalog, so its file stays, and
    /// what the catalog doesn't hold of it moves to a new chunk ([`WorkerLoop::rehome`]).
    fn reject(&mut self, mut closed: Closed, why: &str) {
        let mut rest: Vec<Closed> = Vec::new();
        let queued = std::mem::take(&mut self.uncommitted);
        for c in queued {
            if c.path == closed.path { rest.push(c) } else { self.uncommitted.push_back(c) }
        }
        if closed.more && self.flusher.open_path(closed.stream) == Some(closed.path.as_path()) {
            self.refused = Some(closed.path.clone());
            #[cfg(feature = "ffmpeg")]
            if let Err(e) = self.drain(closed.stream, crate::service::now_utc_ns()) {
                tracing::warn!("draining stream {}'s encoder failed: {e}", closed.stream.0);
            }
            match self.flusher.close(closed.stream) {
                Ok(Some(last)) => self.finish(last),
                Ok(None) => {}
                Err(e) => tracing::warn!("closing {} failed: {e}", closed.path.display()),
            }
            self.refused = None;
            rest.append(&mut self.gathered);
        }
        for c in &rest {
            closed.absorbed.extend_from_slice(&c.absorbed);
        }
        if closed.first_frame > 0 {
            tracing::warn!(
                "the broker refused a part of chunk {} for good ({why}); its {} uncommitted records \
                 move to a new chunk",
                closed.path.display(),
                closed.absorbed.len()
            );
            self.counts.rehomed += 1;
            match self.rehome(&closed.path, closed.stream, closed.first_frame as usize) {
                Ok(Some(moved)) => self.finish(moved),
                Ok(None) => self.out.rejected(&closed, &closed.path),
                Err(e) => {
                    tracing::warn!("moving the records of {} failed: {e}", closed.path.display());
                    self.out.rejected(&closed, &closed.path);
                }
            }
            return;
        }
        let aside = closed.path.with_extension("rejected");
        if let Err(e) = std::fs::rename(&closed.path, &aside) {
            tracing::warn!("setting {} aside failed: {e}", closed.path.display());
        }
        tracing::error!(
            "the broker refused chunk {} for good ({why}); set aside as {}",
            closed.path.display(),
            aside.display()
        );
        self.counts.rejected += 1;
        for id in &closed.absorbed {
            self.recovered_ids.remove(id);
        }
        self.out.rejected(&closed, &aside);
    }

    /// A recovered chunk a purge took from the catalog while the last run was down: the catalog would refuse the
    /// whole file for good (what the chunk had absorbed is gone), and setting it aside would take the records that
    /// arrived after the chunk's last part with it. Those, from record `keep` on, move to a new chunk and are
    /// committed, as a running worker does with a part refused for a hidden chunk. The file keeps the rest, for the
    /// purge to remove.
    fn rehome_recovered(&mut self, stream: StreamId, r: &Recovered, keep: usize) {
        tracing::warn!(
            "the catalog no longer knows recovered chunk {} (a purge took it); its {} records the catalog still \
             holds as payloads move to a new chunk",
            r.path.display(),
            r.records.len() - keep
        );
        self.counts.rehomed += 1;
        // What stays behind is in no chunk the catalog will show: a payload sent again is not a repeat.
        for record in &r.records[..keep] {
            self.recovered_ids.remove(&record.id);
        }
        match self.rehome(&r.path, stream, keep) {
            Ok(Some(moved)) => self.finish(moved),
            Ok(None) => {}
            Err(e) => tracing::warn!("moving the records of {} failed: {e}", r.path.display()),
        }
    }

    /// Moves the records of `stream`'s chunk file at `path` from record `keep` on into a new chunk file beside it,
    /// and returns that as a whole chunk to commit; `None` when a purge cut every one of them.
    fn rehome(&mut self, path: &Path, stream: StreamId, keep: usize) -> Result<Option<Closed>, TierError> {
        let (format, _) = *self.formats.get(&stream).ok_or(TierError::UnknownStream(stream))?;
        let shape = self.sources.get(&stream).ok_or(TierError::UnknownStream(stream))?.clone();
        // The shape the flusher wrote under: a raw camera's is its H.264.
        let shape = StreamShape { chunk_format: format, ..shape };
        let created_ns = crate::service::now_utc_ns();
        // A name the flusher never gives: its counter starts at 1.
        let to = path.with_file_name(format!("{}-{created_ns}-0.cnk", stream.0));
        let records = crate::chunk::split_off(path, keep, &to, created_ns)?;
        if records.is_empty() {
            return Ok(None);
        }
        let chunk = new_chunk(&to, stream, &shape, &records, created_ns)?;
        let absorbed: Vec<Uuid> = crate::flush::catalog_records(&records).iter().map(|r| r.id).collect();
        let provenance = crate::flush::provenance_of(&absorbed, &crate::chunk::read_provenance(&to)?);
        Ok(Some(Closed {
            path: to,
            stream,
            records,
            chunk,
            first_frame: 0,
            more: false,
            absorbed,
            durable: Vec::new(),
            provenance,
            stepped_back: false,
        }))
    }

    fn retry(&mut self) {
        while let Some(c) = self.uncommitted.front() {
            match self.out.commit(c) {
                Ok(()) => {
                    self.uncommitted.pop_front();
                    self.counts.committed += 1;
                }
                Err(CommitFailure::Retry(e)) => {
                    tracing::warn!("committing {} failed again: {e}", c.path.display());
                    self.counts.commit_retries += 1;
                    return;
                }
                Err(CommitFailure::Reject(e)) => {
                    if let Some(c) = self.uncommitted.pop_front() {
                        self.reject(c, &e);
                    }
                }
            }
        }
    }

    fn report(&mut self, ids: &[Uuid]) {
        if ids.is_empty() {
            return;
        }
        match self.out.flushed(ids) {
            Ok(()) => self.counts.flushed += ids.len() as u64,
            Err(e) => tracing::warn!("marking {} slots flushed failed: {e}", ids.len()),
        }
    }
}

#[cfg(test)]
mod camera_boundaries {
    use super::*;

    #[derive(Default)]
    struct Kept(Vec<(std::path::PathBuf, continuum_core::HeaderBuf)>);

    impl Outputs for Kept {
        fn flushed(&mut self, _ids: &[Uuid]) -> Result<(), String> {
            Ok(())
        }
        fn commit(&mut self, closed: &Closed) -> Result<(), CommitFailure> {
            self.0.push((closed.path.clone(), closed.chunk.payload.header.clone()));
            Ok(())
        }
    }

    fn source() -> StreamShape {
        StreamShape {
            chunk_format: ChunkFormat::Passthrough,
            layer: continuum_core::DomainLayer::Layer1Raw,
            format: continuum_core::PayloadFormat::EncodedVideo,
            shape_name: "MJPEG".to_owned(),
            shape_version: 1,
            pcm: None,
            raw: None,
        }
    }

    #[test]
    fn a_capture_epoch_reset_also_splits_a_rate_only_restart() {
        for seqs in [[0, 0], [31, 0], [u64::MAX, 0]] {
            let dir = tempfile::tempdir().unwrap();
            let mut worker = WorkerLoop::new(dir.path(), FlushSettings::default(), Kept::default()).unwrap();
            let stream = StreamId(1);
            worker.declare(stream, source());
            for (i, device_seq) in seqs.into_iter().enumerate() {
                let at = u64::try_from(i).unwrap() * 33_333_333;
                let p = Payload {
                    id: Uuid(uuid::Uuid::now_v7().into_bytes()),
                    stream,
                    start_ns: at,
                    end_ns: at,
                    seq: 0,
                    bytes: b"legacy opaque JPEG".to_vec(),
                    pcm_rate_hz: 0,
                    provenance: None,
                };
                let info = CameraFrameInfo { width: 64, height: 48, stride: 0, flip: false, device_seq };
                worker.video_payload(&p, source(), info, at).unwrap();
            }
            worker.shutdown().unwrap();
            assert_eq!(worker.outputs_mut().0.len(), 2, "capture sequences {seqs:?} must split");
        }
    }
    #[cfg(not(feature = "ffmpeg"))]
    #[test]
    fn raw_fallback_chunks_preserve_each_native_layout_after_compaction() {
        let dir = tempfile::tempdir().unwrap();
        let mut worker = WorkerLoop::new(dir.path(), FlushSettings::default(), Kept::default()).unwrap();
        let stream = StreamId(1);
        for (index, (width, height, flip)) in
            [(64, 48, false), (128, 96, true), (32, 24, false)].into_iter().enumerate()
        {
            let stride = width * 2 + 8;
            let mut shape = source();
            shape.chunk_format = ChunkFormat::Lz4;
            shape.format = continuum_core::PayloadFormat::RawBytes;
            shape.shape_name = "UYVY".to_owned();
            shape.raw = crate::raw::RawLayout::of("UYVY", width, height, stride);
            let mut bytes = vec![0; (stride * height) as usize];
            for row in bytes.chunks_exact_mut(stride as usize) {
                for pixel in row[..(width * 2) as usize].chunks_exact_mut(4) {
                    pixel.copy_from_slice(&[128, 80, 128, 80]);
                }
            }
            let at = index as u64 * 33_333_333;
            let payload = Payload {
                id: Uuid(uuid::Uuid::now_v7().into_bytes()),
                stream,
                start_ns: at,
                end_ns: at,
                seq: 0,
                bytes,
                pcm_rate_hz: 0,
                provenance: None,
            };
            worker
                .video_payload(
                    &payload,
                    shape,
                    CameraFrameInfo { width, height, stride, flip, device_seq: index as u64 },
                    at,
                )
                .unwrap();
        }
        worker.shutdown().unwrap();
        let chunks = &worker.outputs_mut().0;
        assert_eq!(chunks.len(), 3);
        for ((path, header), (width, height, flip)) in
            chunks.iter().zip([(64, 48, false), (128, 96, true), (32, 24, false)])
        {
            let capture = header.header().capture().unwrap();
            assert_eq!(
                (capture.width(), capture.height(), capture.stride(), capture.flip()),
                (width, height, width * 2 + 8, flip)
            );
            assert_eq!(crate::ChunkReader::open(path).unwrap().flip(), flip);
            let frames = crate::hydrate::hydrate(path, header.start_ns(), header.end_ns()).unwrap();
            assert_eq!(frames.len(), 1);
            let bytes = &frames[0].bytes;
            assert_eq!(bytes.len(), ((width * 2 + 8) * height) as usize, "the frame comes back at its own stride");
            assert!(bytes.chunks_exact((width * 2 + 8) as usize).all(|row| row[..4] == [128, 80, 128, 80]));
        }
    }
    #[test]
    fn recovered_raw_fallback_cannot_be_committed_under_a_new_capture_layout() {
        for encoded_restart in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let stream = StreamId(1);
            let hour = crate::layout::hour_dir(dir.path(), stream, 0);
            crate::layout::make_hour_dir(dir.path(), &hour).unwrap();
            let path = hour.join("1-0-1.cnk");
            let id = Uuid([19; 16]);
            let mut writer = crate::ChunkWriter::create(&path, stream, ChunkFormat::Lz4, 0).unwrap();
            writer.append_payload(id, 10, 10, 0, &crate::codecs::lz4_compress(&vec![128; 64 * 48 * 2])).unwrap();
            writer.sync().unwrap();
            drop(writer);
            let recovered = crate::recover(dir.path()).unwrap();
            let original = std::fs::read(&path).unwrap();
            let mut shape = source();
            shape.chunk_format = ChunkFormat::Lz4;
            shape.format = continuum_core::PayloadFormat::RawBytes;
            shape.shape_name = "UYVY".to_owned();
            shape.raw = crate::raw::RawLayout::of("UYVY", 128, 96, 256);
            if encoded_restart {
                shape = source();
            }
            let mut worker = WorkerLoop::new(dir.path(), FlushSettings::default(), Kept::default()).unwrap();
            worker.recovered(recovered);
            assert!(worker.is_recovered(id));
            worker.declare(stream, shape.clone());
            assert!(!worker.is_recovered(id), "ambiguous bytes must not acknowledge replay as recovered");
            assert!(worker.recovery_error(stream).is_none());
            assert!(worker.is_quarantined(id));
            let payload = Payload {
                id,
                stream,
                start_ns: 10,
                end_ns: 10,
                seq: 0,
                bytes: vec![],
                pcm_rate_hz: 0,
                provenance: None,
            };
            assert_eq!(worker.payload(&payload, 10).unwrap(), Appended::Quarantined);
            worker.shutdown().unwrap();
            assert!(worker.outputs().0.is_empty());
            assert_eq!(worker.counts().flushed, 0);
            let aside = std::fs::read_dir(&hour)
                .unwrap()
                .map(|e| e.unwrap().path())
                .find(|p| p.extension().is_some_and(|ext| ext == "layout-unknown"))
                .unwrap();
            assert_eq!(std::fs::read(&aside).unwrap(), original);
            let mut restarted = WorkerLoop::new(dir.path(), FlushSettings::default(), Kept::default()).unwrap();
            restarted.declare(stream, shape);
            assert!(restarted.recovery_error(stream).is_none());
            assert!(restarted.is_quarantined(id));
            assert_eq!(restarted.payload(&payload, 10).unwrap(), Appended::Quarantined);
            assert_eq!(restarted.counts().flushed, 0);
            assert!(restarted.outputs().0.is_empty());
            let bytes = if encoded_restart {
                let mut jpeg = Vec::new();
                jpeg_encoder::Encoder::new(&mut jpeg, 90)
                    .encode(&vec![128; 128 * 96 * 3], 128, 96, jpeg_encoder::ColorType::Rgb)
                    .unwrap();
                jpeg
            } else {
                vec![128; 128 * 96 * 2]
            };
            let fresh = Payload { id: Uuid([20; 16]), bytes, start_ns: 20, end_ns: 20, ..payload.clone() };
            assert_eq!(restarted.payload(&fresh, 20).unwrap(), Appended::Appended);
            let other = StreamId(2);
            restarted.declare(other, source());
            let fresh_other = Payload { id: Uuid([21; 16]), stream: other, bytes: b"JPEG".to_vec(), ..fresh };
            assert_eq!(restarted.payload(&fresh_other, 21).unwrap(), Appended::Appended);
            restarted.shutdown().unwrap();
            assert_eq!(restarted.outputs().0.len(), 2, "later same-camera and other-stream chunks continue");
        }
    }
}
