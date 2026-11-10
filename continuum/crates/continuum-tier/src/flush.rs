//! The flush from Tier 0 to Tier 1: every payload is appended to its stream's open chunk as it lands. A camera's
//! chunk is one file for `chunk_ns` (a minute), and the catalog hears about its new records every `part_ns` (2
//! seconds), so the ring only has to cover that lag, never a chunk (docs/CONTINUUM.md, "Video chunks").
//!
//! The flusher knows nothing about sockets or shared memory. The worker feeds it each payload's header fields and
//! bytes (read from the slot) and commits each [`Closed`] it hands back: a chunk's first part with
//! [`continuum_catalog::Catalog::commit_chunk`], which compacts the absorbed frames' headers into the chunk's
//! (docs/CONTINUUM.md, "Catalog"), and each later part with [`continuum_catalog::Catalog::extend_chunk`], which
//! does the same for the frames the part adds. A part is cut on a keyframe, or once an H.264 stream has gone
//! quiet, and after a sync, so the catalog never names bytes a power cut could drop.
//! A microphone's chunk is committed once, when it closes, since it becomes Opus then.
//!
//! The `flushed` contract: a slot is marked `flushed` only once its record is durable. [`Flusher::append`] writes
//! the record straight into the chunk file (a killed worker loses none of it, the page cache has it), but a power
//! loss can still drop it until the file is synced, so the worker marks nothing on append. Instead it calls
//! [`Flusher::sync_due`] on a short cadence (`sync_ns`, 200 ms by default): that syncs every chunk with records
//! waiting and returns the ids whose records are now on disk, and only those slots are marked `flushed`. One fsync
//! per chunk every few hundred milliseconds instead of one per frame keeps 30 fps times several cameras off the
//! disk; the Tier 0 ring is 10 seconds deep, so the lag costs nothing. A chunk's close syncs it too and reports the
//! ids it made durable in [`Closed::durable`].
//!
//! A chunk closes when its time is up, when it reaches the catalog's frame limit, or when a payload arrives whose
//! time is before the last record's (a retransmit, an out-of-order ingest): the catalog needs a chunk's frames in
//! time order, so the late payload starts a new chunk, whose span may overlap the old one's, which the catalog
//! allows.
//!
//! A chunk's id is derived from its file's name ([`crate::layout::chunk_id`]), so every part names the same chunk
//! and a recovery after a crash finds what the catalog already holds of it.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use continuum_catalog::{ChunkFrame, MAX_CHUNK_FRAMES, NewChunk, NewPayload};
use continuum_core::{DomainLayer, HeaderBuf, HeaderSpec, PayloadFormat, StorageLocator, StorageTier, StreamId, Uuid};

use crate::TierError;
use crate::chunk::{ChunkError, ChunkFormat, ChunkWriter, Record};
use crate::worker::CameraFrameInfo;

/// How the flusher cuts and syncs chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlushSettings {
    /// How long a chunk file stays open, in nanoseconds, for everything but a microphone: a camera writes one file a
    /// minute, closed on the first keyframe after this.
    pub chunk_ns: u64,
    /// How often an open chunk's new records are committed, in nanoseconds: what bounds the time a frame is in Tier
    /// 0 alone, so it stays well inside the 10 s ring. At or above `chunk_ns`, a chunk is committed once, at close.
    pub part_ns: u64,
    /// How long a microphone's chunk stays open (PCM batches, which become Opus at close). Audio has no keyframes,
    /// and every chunk is kept forever with its catalog rows, so it's longer. Tier 1 readers such as
    /// `room_audio` see a line once its chunk closes, so it stays short enough for a fresh line to play soon.
    pub audio_chunk_ns: u64,
    /// How often [`Flusher::sync_due`] syncs a chunk with records waiting, in nanoseconds. Shorter means less is
    /// lost to a power cut and slots are reclaimable sooner; longer means fewer fsyncs. 100 to 250 ms is the range
    /// that makes sense against a 10 s ring.
    pub sync_ns: u64,
}

impl FlushSettings {
    /// How long a chunk of `format` stays open.
    #[must_use]
    pub fn open_ns(&self, format: ChunkFormat) -> u64 {
        if matches!(format, ChunkFormat::PcmBatch | ChunkFormat::Opus) { self.audio_chunk_ns } else { self.chunk_ns }
    }

    /// How often an open chunk of `format` commits a part; `None` for a microphone's, which is committed once, at
    /// its close (it becomes Opus then, and a chunk has one format).
    #[must_use]
    pub fn part_ns(&self, format: ChunkFormat) -> Option<u64> {
        (!matches!(format, ChunkFormat::PcmBatch | ChunkFormat::Opus)).then_some(self.part_ns)
    }
}

impl Default for FlushSettings {
    fn default() -> Self {
        Self { chunk_ns: 60_000_000_000, part_ns: 2_000_000_000, audio_chunk_ns: 15_000_000_000, sync_ns: 200_000_000 }
    }
}

/// What the flusher must know about a stream to write its chunks and their headers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamShape {
    /// How the chunk stores the stream's payloads.
    pub chunk_format: ChunkFormat,
    /// The payloads' domain layer, copied onto the chunk's header.
    pub layer: DomainLayer,
    /// The payloads' format, copied onto the chunk's header.
    pub format: PayloadFormat,
    /// The payloads' shape name, copied onto the chunk's header.
    pub shape_name: String,
    /// The shape's version.
    pub shape_version: u32,
    /// For a [`ChunkFormat::PcmBatch`] stream, its ring records' layout.
    pub pcm: Option<crate::pcm::PcmLayout>,
    /// For a raw camera stream (UYVY, YUYV, or NV12 frames), its frames' layout, from the first frame's capture
    /// info: what its encoder is opened for.
    pub raw: Option<crate::raw::RawLayout>,
}

/// One payload as it lands in Tier 0.
#[derive(Clone, Copy, Debug)]
pub struct Incoming<'a> {
    /// The payload's id.
    pub id: Uuid,
    /// Its stream.
    pub stream: StreamId,
    /// Its start, UTC nanoseconds.
    pub start_ns: u64,
    /// Its end; equal to `start_ns` for an instant.
    pub end_ns: u64,
    /// Its sequence number.
    pub seq: u32,
    /// Its bytes, already in the chunk's format (the caller applies [`crate::codecs`] for LZ4 or FLAC streams).
    pub bytes: &'a [u8],
    /// Decoding can start here. Always true but for an [`ChunkFormat::H264Au`] stream's P-frames: a chunk of one
    /// only ever starts on a keyframe, so it decodes on its own.
    pub key: bool,
}

/// How long an [`ChunkFormat::H264Au`] stream must be quiet before [`Flusher::rotate`] closes its chunk on time: a
/// stream still sending closes on its next keyframe instead (at most a GOP, a second, later), so its next chunk starts
/// on one.
pub const QUIET_NS: u64 = 1_000_000_000;

/// What a `CommitChunk` to the broker carries, borrowed from a [`Closed`]: the chunk's own header and location, the
/// frames this commit adds, and the Tier 0 payloads behind them, in the frames' order.
#[derive(Clone, Copy, Debug)]
pub struct ChunkCommit<'a> {
    /// The chunk's header (on the frames' stream, spanning every frame so far, no parents).
    pub header: &'a HeaderBuf,
    /// Where the chunk file is, and how long it is now.
    pub location: &'a StorageLocator,
    /// The index of `frames[0]` in the whole chunk: 0 for a chunk's first commit.
    pub first_frame: u32,
    /// One entry per absorbed payload, in order.
    pub frames: &'a [ChunkFrame],
    /// The absorbed payloads' ids, in the same order.
    pub absorbed: &'a [Uuid],
}

/// What the flusher hands over for the catalog: a chunk it closed, or a part of one (the records since the chunk's
/// last commit; `first_frame` and `more` say which). A chunk committed once is one of these with `first_frame` 0
/// and `more` false.
#[derive(Debug)]
pub struct Closed {
    /// The chunk's file.
    pub path: PathBuf,
    /// Its stream.
    pub stream: StreamId,
    /// The records this commit adds, in file order.
    pub records: Vec<Record>,
    /// The chunk's payload (its header and location as they are now) and the frames this commit adds.
    pub chunk: NewChunk,
    /// The index of this commit's first frame in the whole chunk: 0 commits the chunk, anything else extends it.
    pub first_frame: u32,
    /// The chunk's file is still open: more parts follow.
    pub more: bool,
    /// The Tier 0 payloads it absorbed, in the same order as `chunk.frames` (each record carries its id, so a
    /// recovered chunk has them too).
    pub absorbed: Vec<Uuid>,
    /// The absorbed payloads the close made durable: those not yet reported by [`Flusher::sync_due`]. Mark these
    /// slots `flushed` now.
    pub durable: Vec<Uuid>,
    /// What capture attested about each frame, in `chunk.frames` order ([`Flusher::attest`]); `None` when no
    /// frame's producer said, or for a chunk recovered from its file, which doesn't keep it.
    pub provenance: Option<continuum_catalog::ChunkProvenance>,
    /// These records start before the chunk's last committed frame ([`steps_back`]): the catalog would refuse
    /// them as a part, so they go to a chunk of their own instead. Never set for a whole chunk.
    pub stepped_back: bool,
}

/// Whether a chunk's records from `committed` on start before the latest of the ones before them: the stream's
/// times went back since the chunk's last part (capture started a network camera's clock fit over, say). The
/// catalog keeps a chunk's frame index in time order across parts, so it refuses such a part for good. B-frames
/// don't do this: they reorder inside a GOP, and a part is cut only before a keyframe.
#[must_use]
pub fn steps_back(records: &[Record], committed: usize) -> bool {
    let committed = committed.min(records.len());
    let latest = records[..committed].iter().map(|r| r.start_ns).max();
    let earliest = records[committed..].iter().map(|r| r.start_ns).min();
    latest.zip(earliest).is_some_and(|(latest, earliest)| earliest < latest)
}

impl Closed {
    /// The chunk as a `CommitChunk` carries it, for the broker.
    ///
    /// # Panics
    ///
    /// Never: a closed chunk always has a location ([`new_chunk`] sets it).
    #[must_use]
    pub fn to_commit(&self) -> ChunkCommit<'_> {
        ChunkCommit {
            header: &self.chunk.payload.header,
            location: self.chunk.payload.location.as_ref().expect("a closed chunk has a location"),
            first_frame: self.first_frame,
            frames: &self.chunk.frames,
            absorbed: &self.absorbed,
        }
    }
}

struct Open {
    writer: ChunkWriter,
    /// The chunk's id, the same for every part ([`crate::layout::chunk_id`]).
    id: Uuid,
    /// How many of the writer's records earlier parts handed over.
    committed: usize,
    /// When the last part was cut (or the chunk was opened).
    part_ns: u64,
    /// Appended but not yet synced.
    pending: Vec<Uuid>,
    last_sync_ns: u64,
    /// When the last payload was appended.
    last_append_ns: u64,
}

/// Writes each stream's payloads into its open chunk, syncs them on a cadence, and closes chunks on time.
pub struct Flusher {
    dir: PathBuf,
    settings: FlushSettings,
    shapes: BTreeMap<StreamId, StreamShape>,
    camera_layouts: BTreeMap<StreamId, CameraFrameInfo>,
    open: BTreeMap<StreamId, Open>,
    /// Makes chunk file names unique even when two open at the same nanosecond.
    counter: u64,
    /// Chunks closed just before a new chunk's create failed (see [`Flusher::parked`]).
    parked: Vec<Closed>,
    /// The hour directories made durable this run (`layout::make_hour_dir`, once each).
    synced: std::collections::BTreeSet<PathBuf>,
    /// Capture provenance by stream and payload id, until the chunk holding the payload closes.
    attested: BTreeMap<StreamId, BTreeMap<Uuid, continuum_catalog::FrameProvenance>>,
}

impl Flusher {
    /// A flusher writing chunks under `dir` (created if missing).
    ///
    /// # Errors
    ///
    /// An I/O error creating the directory.
    pub fn new(dir: &Path, settings: FlushSettings) -> Result<Self, TierError> {
        fs::create_dir_all(dir)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            settings,
            shapes: BTreeMap::new(),
            camera_layouts: BTreeMap::new(),
            open: BTreeMap::new(),
            counter: 0,
            parked: Vec::new(),
            synced: std::collections::BTreeSet::new(),
            attested: BTreeMap::new(),
        })
    }

    /// Records what capture attested about payload `id` of `stream`, for the chunk it lands in. A raw camera's
    /// frames reach the chunk later than they arrive (the encoder holds a few), so this is by id, not by record.
    pub fn attest(&mut self, stream: StreamId, id: Uuid, provenance: continuum_catalog::FrameProvenance) {
        self.attested.entry(stream).or_default().insert(id, provenance);
    }

    /// The directory chunks are written under.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Declares a stream. Payloads of an undeclared stream are refused.
    pub fn declare(&mut self, stream: StreamId, shape: StreamShape) {
        self.shapes.insert(stream, shape);
    }

    /// The current camera layout, installed only after the old chunk has closed at a source boundary.
    pub(crate) fn camera_layout(&mut self, stream: StreamId, info: CameraFrameInfo) {
        self.camera_layouts.insert(stream, info);
    }

    /// Streams with an open chunk.
    #[must_use]
    pub fn open_streams(&self) -> Vec<StreamId> {
        self.open.keys().copied().collect()
    }

    /// The file of a stream's open chunk, if it has one.
    #[must_use]
    pub fn open_path(&self, stream: StreamId) -> Option<&Path> {
        self.open.get(&stream).map(|o| o.writer.path())
    }

    /// Closes a stream's current chunk before its source layout changes. An idle stream has nothing to close.
    ///
    /// # Errors
    ///
    /// A chunk or I/O error. The caller must leave the old stream shape in place until this succeeds.
    pub fn close(&mut self, stream: StreamId) -> Result<Option<Closed>, TierError> {
        if self.open.contains_key(&stream) { self.close_stream(stream) } else { Ok(None) }
    }

    /// Payloads appended but not yet durable, across every open chunk.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.open.values().map(|o| o.pending.len()).sum()
    }

    /// A new chunk at `path` in its hour directory `hour`: the directory made and synced into its parents once a run,
    /// and the chunk's own entry synced into it, so neither is lost in a crash. A purge may have removed the hour's
    /// directory as its last chunk went (`layout::prune_hour_dir`) just before: then it's made again, once.
    fn create_in(
        &mut self,
        hour: &Path,
        path: &Path,
        stream: StreamId,
        format: ChunkFormat,
        now_ns: u64,
    ) -> Result<ChunkWriter, ChunkError> {
        for again in [false, true] {
            if again || !self.synced.contains(hour) {
                crate::layout::make_hour_dir(&self.dir, hour)?;
                self.synced.insert(hour.to_path_buf());
            }
            match ChunkWriter::create(path, stream, format, now_ns) {
                Ok(writer) => {
                    fs::File::open(hour)?.sync_all()?;
                    return Ok(writer);
                }
                Err(ChunkError::Io(e)) if !again && e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            }
        }
        unreachable!("the second try returns")
    }

    /// Appends a payload to its stream's chunk, opening one if needed and first closing one that is past its time,
    /// full, ahead of this payload in time, or due a part its records since the last one can't be ([`steps_back`]). Returns the record and the chunk closed on the way, if any. The
    /// payload is in the file but not yet durable: its slot stays unflushed until [`Flusher::sync_due`] or a close
    /// reports its id.
    ///
    /// # Errors
    ///
    /// [`TierError::UnknownStream`] for an undeclared stream, or a chunk or I/O error. When the new chunk can't be
    /// created after an old one closed, the closed chunk is parked rather than lost: see [`Flusher::parked`].
    pub fn append(&mut self, p: Incoming<'_>, now_ns: u64) -> Result<(Record, Option<Closed>), TierError> {
        let shape = self.shapes.get(&p.stream).ok_or(TierError::UnknownStream(p.stream))?.clone();
        let mut closed = None;
        if let Some(open) = self.open.get(&p.stream) {
            // An H.264 stream's chunk closes only before a keyframe: its next chunk has to start on one.
            let aged =
                now_ns.saturating_sub(open.writer.created_ns()) >= self.settings.open_ns(shape.chunk_format) && p.key;
            let full = open.writer.records().len() >= MAX_CHUNK_FRAMES;
            // Native encoded cameras arrive in decode order; their B-frame PTS can legitimately regress.
            let backwards = !(shape.chunk_format == ChunkFormat::H264Au && shape.shape_name == "H264AnnexB")
                && open.writer.records().last().is_some_and(|last| p.start_ns < last.start_ns);
            // A part is cut only before a keyframe, so its frames never present before the last part's (B-frames
            // reorder only inside a GOP).
            let part = self
                .settings
                .part_ns(shape.chunk_format)
                .is_some_and(|every| now_ns.saturating_sub(open.part_ns) >= every)
                && p.key
                && open.writer.records().len() > open.committed;
            // The records since the last part start before it: no part of this chunk can take them. The chunk
            // closes here, on a keyframe, and the worker moves them to a chunk of their own.
            let stepped = part && steps_back(open.writer.records(), open.committed);
            if aged || full || backwards || stepped {
                closed = self.close_stream(p.stream)?;
            } else if part {
                closed = Some(self.cut_part(p.stream, now_ns)?);
            }
        }
        if !self.open.contains_key(&p.stream) {
            self.counter += 1;
            // In its stream's UTC-hour directory, named as before.
            let hour = crate::layout::hour_dir(&self.dir, p.stream, now_ns);
            let path = hour.join(format!("{}-{}-{}.cnk", p.stream.0, now_ns, self.counter));
            let id = crate::layout::chunk_id(p.stream, now_ns, self.counter);
            let mut writer = match self.create_in(&hour, &path, p.stream, shape.chunk_format, now_ns) {
                Ok(w) => w,
                Err(e) => {
                    // The chunk closed above is finished and still has to reach the catalog: recovery skips finished
                    // files, so it is parked for `parked`, `rotate`, or `close_all` to hand back.
                    self.parked.extend(closed.take());
                    return Err(e.into());
                }
            };
            writer.set_flip(self.camera_layouts.get(&p.stream).is_some_and(|c| c.flip))?;
            self.open.insert(
                p.stream,
                Open {
                    writer,
                    id,
                    committed: 0,
                    part_ns: now_ns,
                    pending: Vec::new(),
                    last_sync_ns: now_ns,
                    last_append_ns: now_ns,
                },
            );
        }
        let open = self.open.get_mut(&p.stream).expect("just opened");
        // Ahead of the frame's own record, so a crash between the two leaves no frame claiming it (finalize trims it).
        if let Some(&provenance) = self.attested.get(&p.stream).and_then(|a| a.get(&p.id)) {
            open.writer.append_provenance(p.id, provenance)?;
        }
        let record = open.writer.append_payload(p.id, p.start_ns, p.end_ns, p.seq, p.bytes)?;
        open.pending.push(p.id);
        open.last_append_ns = now_ns;
        Ok((record, closed))
    }

    /// Syncs every open chunk that has records waiting and was last synced `sync_ns` or more ago, and returns the
    /// ids whose records are now durable. Call it on each tick of the worker's loop; mark the returned slots
    /// `flushed`.
    ///
    /// # Errors
    ///
    /// An I/O error.
    pub fn sync_due(&mut self, now_ns: u64) -> Result<Vec<Uuid>, TierError> {
        let mut durable = Vec::new();
        for o in self.open.values_mut() {
            if !o.pending.is_empty() && now_ns.saturating_sub(o.last_sync_ns) >= self.settings.sync_ns {
                o.writer.sync()?;
                o.last_sync_ns = now_ns;
                durable.append(&mut o.pending);
            }
        }
        Ok(durable)
    }

    /// Syncs every open chunk now, whatever the cadence, and returns the ids made durable.
    ///
    /// # Errors
    ///
    /// An I/O error.
    pub fn sync_all(&mut self) -> Result<Vec<Uuid>, TierError> {
        let mut durable = Vec::new();
        for o in self.open.values_mut() {
            if !o.pending.is_empty() {
                o.writer.sync()?;
                durable.append(&mut o.pending);
            }
        }
        Ok(durable)
    }

    /// The streams [`Flusher::rotate`] would act on now: a chunk past its time, or one with records a part is due
    /// for, and for an [`ChunkFormat::H264Au`] stream also quiet for [`QUIET_NS`] (its encoder has to be drained
    /// first; see [`crate::worker`]). A stream still sending H.264 is cut in [`Flusher::append`], on its keyframes.
    #[must_use]
    pub fn due(&self, now_ns: u64) -> Vec<StreamId> {
        self.open
            .iter()
            .filter(|(_, o)| {
                o.writer.format() != ChunkFormat::H264Au || now_ns.saturating_sub(o.last_append_ns) >= QUIET_NS
            })
            .filter(|(_, o)| self.closes(o, now_ns) || self.parts(o, now_ns))
            .map(|(s, _)| *s)
            .collect()
    }

    fn closes(&self, o: &Open, now_ns: u64) -> bool {
        now_ns.saturating_sub(o.writer.created_ns()) >= self.settings.open_ns(o.writer.format())
    }

    /// Whether a part is due from `o` on the timer. Not from records that stepped back ([`steps_back`]): the
    /// catalog would refuse the part and the chunk would close wherever the stream then was. A quiet network camera
    /// is often one whose link stalled, which is also when capture starts its clock fit over; those records wait for
    /// the stream's next keyframe ([`Flusher::append`] closes the chunk there) or for the chunk's own close.
    fn parts(&self, o: &Open, now_ns: u64) -> bool {
        o.writer.records().len() > o.committed
            && self.settings.part_ns(o.writer.format()).is_some_and(|every| now_ns.saturating_sub(o.part_ns) >= every)
            && !steps_back(o.writer.records(), o.committed)
    }

    /// Closes every chunk [`Flusher::due`] names that is open for `chunk_ns` or longer, and cuts a part from the
    /// others it names.
    ///
    /// # Errors
    ///
    /// A chunk or I/O error.
    pub fn rotate(&mut self, now_ns: u64) -> Result<Vec<Closed>, TierError> {
        let due = self.due(now_ns);
        let mut out = std::mem::take(&mut self.parked);
        for s in due {
            let closes = self.open.get(&s).is_some_and(|o| self.closes(o, now_ns));
            let done = if closes { self.close_stream(s) } else { self.cut_part(s, now_ns).map(Some) };
            match done {
                Ok(c) => out.extend(c),
                // What closed so far stays parked for the next call, not lost with the error.
                Err(e) => {
                    self.parked = out;
                    return Err(e);
                }
            }
        }
        Ok(out)
    }

    /// Chunks closed by an [`Flusher::append`] whose new chunk couldn't be created: finished files the catalog
    /// doesn't know yet. [`Flusher::rotate`] and [`Flusher::close_all`] hand them back too, so a worker that commits
    /// what those return needs nothing more; one that stops on the error collects them here first.
    #[must_use]
    pub fn parked(&mut self) -> Vec<Closed> {
        std::mem::take(&mut self.parked)
    }

    /// Closes every open chunk (shutdown), after any parked ones.
    ///
    /// # Errors
    ///
    /// A chunk or I/O error.
    pub fn close_all(&mut self) -> Result<Vec<Closed>, TierError> {
        let streams: Vec<StreamId> = self.open.keys().copied().collect();
        let mut out = std::mem::take(&mut self.parked);
        for s in streams {
            match self.close_stream(s) {
                Ok(c) => out.extend(c),
                Err(e) => {
                    self.parked = out;
                    return Err(e);
                }
            }
        }
        Ok(out)
    }

    /// Closes a stream's chunk and hands over what no part did yet: its last part, or the whole chunk. `None` when
    /// every record was in a part already (the file gains only its footer).
    fn close_stream(&mut self, stream: StreamId) -> Result<Option<Closed>, TierError> {
        let open = self.open.remove(&stream).ok_or(TierError::UnknownStream(stream))?;
        let shape = self.shapes.get(&stream).ok_or(TierError::UnknownStream(stream))?.clone();
        let path = open.writer.path().to_path_buf();
        let created_ns = open.writer.created_ns();
        let all = open.writer.close()?;
        if all.len() == open.committed && open.committed > 0 {
            debug_assert!(open.pending.is_empty(), "a part's cut syncs everything it hands over");
            return Ok(None);
        }
        let camera = self.camera_layouts.get(&stream).copied();
        let chunk = part_chunk(&path, open.id, stream, &shape, &all, open.committed, created_ns, camera)?;
        let records = all[open.committed..].to_vec();
        let absorbed: Vec<Uuid> = catalog_records(&records).iter().map(|r| r.id).collect();
        let provenance = self.take_attested(stream, &absorbed);
        let first_frame = u32::try_from(open.committed).unwrap_or(u32::MAX);
        let stepped_back = steps_back(&all, open.committed);
        Ok(Some(Closed {
            path,
            stream,
            records,
            chunk,
            first_frame,
            more: false,
            absorbed,
            durable: open.pending,
            provenance,
            stepped_back,
        }))
    }

    /// Hands over an open chunk's records since its last part, synced first: the chunk stays open.
    fn cut_part(&mut self, stream: StreamId, now_ns: u64) -> Result<Closed, TierError> {
        let shape = self.shapes.get(&stream).ok_or(TierError::UnknownStream(stream))?.clone();
        let camera = self.camera_layouts.get(&stream).copied();
        let open = self.open.get_mut(&stream).ok_or(TierError::UnknownStream(stream))?;
        open.writer.sync()?;
        open.last_sync_ns = now_ns;
        open.part_ns = now_ns;
        let durable = std::mem::take(&mut open.pending);
        let (path, id, from) = (open.writer.path().to_path_buf(), open.id, open.committed);
        let all = open.writer.records();
        let chunk = part_chunk(&path, id, stream, &shape, all, from, open.writer.created_ns(), camera)?;
        let records = all[from..].to_vec();
        open.committed = all.len();
        let absorbed: Vec<Uuid> = catalog_records(&records).iter().map(|r| r.id).collect();
        let provenance = self.take_attested(stream, &absorbed);
        let first_frame = u32::try_from(from).unwrap_or(u32::MAX);
        Ok(Closed {
            path,
            stream,
            records,
            chunk,
            first_frame,
            more: true,
            absorbed,
            durable,
            provenance,
            stepped_back: false,
        })
    }

    /// The chunk's provenance, one per frame (unknown frames get epoch 0), or `None` when nothing was attested.
    /// Payload ids are UUIDv7s and a stream's payloads reach its chunks in arrival order, so everything up to the
    /// chunk's newest id is either in it or never landed: it all goes, which bounds the map.
    fn take_attested(&mut self, stream: StreamId, ids: &[Uuid]) -> Option<continuum_catalog::ChunkProvenance> {
        let attested = self.attested.get_mut(&stream)?;
        let provenance = provenance_of(ids, attested);
        if let Some(&newest) = ids.iter().max() {
            *attested = attested.split_off(&newest);
            attested.remove(&newest);
        }
        provenance
    }
}

/// A chunk's provenance from what is known by payload id, one row per id in `ids` (frame order; unknown frames get
/// epoch 0), or `None` when none is known.
pub(crate) fn provenance_of(
    ids: &[Uuid],
    known: &BTreeMap<Uuid, continuum_catalog::FrameProvenance>,
) -> Option<continuum_catalog::ChunkProvenance> {
    let rows: Vec<Option<continuum_catalog::FrameProvenance>> = ids.iter().map(|id| known.get(id).copied()).collect();
    rows.iter()
        .any(Option::is_some)
        .then(|| continuum_catalog::ChunkProvenance::new(rows.into_iter().map(Option::unwrap_or_default).collect()))
}

/// The catalog indexes presentation time, independently of the physical decoder-order record slots.
/// Keep the same order for frames and absorbed IDs, including after crash recovery and purge rewrites.
pub(crate) fn catalog_record_indexes(records: &[Record]) -> Vec<usize> {
    let mut indexes: Vec<_> = (0..records.len()).collect();
    indexes.sort_by_key(|&index| {
        let record = &records[index];
        (record.start_ns, record.seq, record.offset)
    });
    indexes
}

pub(crate) fn catalog_records(records: &[Record]) -> Vec<&Record> {
    catalog_record_indexes(records).into_iter().map(|index| &records[index]).collect()
}

/// The catalog's chunk payload and frame index for a whole chunk file, under the id its name gives.
///
/// # Errors
///
/// [`TierError::Header`] if the header can't be built (an empty chunk has no span), or an I/O error reading the
/// file's length.
pub fn new_chunk(
    path: &Path,
    stream: StreamId,
    shape: &StreamShape,
    records: &[Record],
    created_ns: u64,
) -> Result<NewChunk, TierError> {
    new_part(path, stream, shape, records, 0, created_ns)
}

/// As [`new_chunk`], for the chunk's records from `from` on (a recovered chunk the catalog holds the start of): the
/// header spans every record, and the frames are those from `from`.
///
/// # Errors
///
/// As [`new_chunk`].
pub fn new_part(
    path: &Path,
    stream: StreamId,
    shape: &StreamShape,
    records: &[Record],
    from: usize,
    created_ns: u64,
) -> Result<NewChunk, TierError> {
    let flip = crate::ChunkReader::open(path)?.flip();
    let camera = shape
        .raw
        .map(|raw| CameraFrameInfo { width: raw.width, height: raw.height, stride: raw.stride, flip, device_seq: 0 })
        .or_else(|| flip.then_some(CameraFrameInfo { width: 0, height: 0, stride: 0, flip, device_seq: 0 }));
    // A file whose name isn't the flusher's (a test fixture) gets an id of its own.
    let id = crate::layout::chunk_id_of(path).unwrap_or_else(|| Uuid(uuid::Uuid::now_v7().into_bytes()));
    part_chunk(path, id, stream, shape, records, from.min(records.len()), created_ns, camera)
}

/// Chunk `id`'s payload as it is with `records` in its file, and the frames of those from `from` on.
#[allow(clippy::too_many_arguments)]
fn part_chunk(
    path: &Path,
    id: Uuid,
    stream: StreamId,
    shape: &StreamShape,
    records: &[Record],
    from: usize,
    created_ns: u64,
    camera: Option<CameraFrameInfo>,
) -> Result<NewChunk, TierError> {
    // The chunk's span is half-open and must reach past its last frame: an instant frame at t covers [t, t + 1), the
    // same rule as Interval::covering.
    let (start_ns, end_ns) = match records.first() {
        Some(first) => {
            let end = records.iter().map(Record::end_exclusive).max().unwrap_or(first.start_ns.saturating_add(1));
            let start = records.iter().map(|r| r.start_ns).min().unwrap_or(first.start_ns);
            (start, end.max(start.saturating_add(1)))
        }
        None => (created_ns, created_ns.saturating_add(1)),
    };
    let mut spec = HeaderSpec::raw(id, stream, start_ns, end_ns);
    spec.layer = shape.layer;
    spec.format = shape.format;
    spec.shape_name.clone_from(&shape.shape_name);
    spec.shape_version = shape.shape_version;
    spec.source_tier = StorageTier::Tier0Hot;
    let camera = camera.or_else(|| {
        shape.raw.map(|raw| CameraFrameInfo {
            width: raw.width,
            height: raw.height,
            stride: raw.stride,
            flip: false,
            device_seq: 0,
        })
    });
    let header = match camera {
        Some(info) => camera_chunk_header(&spec, info)?,
        None => HeaderBuf::build(&spec)?,
    };
    let length = fs::metadata(path)?.len();
    let location = StorageLocator::Chunk {
        tier: StorageTier::Tier1Warm,
        uri: path.to_string_lossy().into_owned(),
        offset: 0,
        length,
    };
    Ok(NewChunk {
        payload: NewPayload { header, location: Some(location), coverage: None },
        frames: catalog_records(&records[from..]).iter().map(|r| r.frame()).collect(),
    })
}

/// Preserve the layout after the individual frame headers are compacted away. This is essential for raw LZ4
/// fallback; encoded video is self-describing, but retaining its flip flag also keeps native readers informed.
fn camera_chunk_header(spec: &HeaderSpec, info: CameraFrameInfo) -> Result<HeaderBuf, TierError> {
    use continuum_core::schema::continuum as fb;
    let mut builder = flatbuffers::FlatBufferBuilder::new();
    let shape = builder.create_string(&spec.shape_name);
    let capture = fb::CaptureInfo::create(
        &mut builder,
        &fb::CaptureInfoArgs {
            width: info.width,
            height: info.height,
            stride: info.stride,
            flip: info.flip,
            ..Default::default()
        },
    );
    let id = fb::Uuid::from(spec.id);
    let header = fb::PayloadHeader::create(
        &mut builder,
        &fb::PayloadHeaderArgs {
            id: Some(&id),
            stream_id: spec.stream.0,
            start_ns: spec.start_ns,
            end_ns: spec.end_ns,
            seq: spec.seq,
            layer: spec.layer,
            format: spec.format,
            shape_name: Some(shape),
            shape_version: spec.shape_version,
            source_tier: spec.source_tier,
            capture: Some(capture),
            ..Default::default()
        },
    );
    fb::finish_payload_header_buffer(&mut builder, header);
    Ok(HeaderBuf::new(builder.finished_data().to_vec())?)
}
