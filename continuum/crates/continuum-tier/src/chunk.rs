//! The chunk file: an append-only container of concatenated payloads with their times, and a frame index.
//!
//! A Tier 1 chunk holds a stretch of one stream's payloads (a minute of a camera, 15 seconds of a microphone) as the
//! producer wrote them (MJPEG frames as JPEG
//! bytes, LZ4 or FLAC or Arrow IPC bytes from [`crate::codecs`]), one record after another. Each record is
//! self-describing (length, times, sequence, a CRC over its header fields and bytes), and every append writes
//! straight to the file, so a worker that is killed loses nothing it appended (the page cache has it) and a power
//! loss loses at most what was appended since the last [`ChunkWriter::sync`] or [`ChunkWriter::close`].
//! [`ChunkReader::open`] recovers every whole record without a footer; a corrupt record is skipped by resyncing on
//! the next record whose CRC checks out, so one flipped byte costs one record. The index (one entry per record)
//! is written as a footer when the chunk is closed, and [`finalize`] writes it for a chunk that was never closed.
//!
//! The catalog keeps the same frame index in its `CHUNKS` table ([`ChunkFrame`]), so a hydrate can read one frame
//! by offset without opening the index at all. A purge that cuts a frame turns its record into a tombstone (same
//! place, same times, no bytes), so frame indexes never move ([`rewrite_without`]).
//!
//! Layout, all integers little-endian:
//!
//! ```text
//! header   "CNK1" | version u16 | flags u16 (bit 0: native camera flip) | stream_id u32 | format tag u32 | created_ns u64        (24 bytes)
//! record   tag u8 ("R" live, "T" tombstone) | len u32 | start_ns u64 | end_ns u64 | seq u32 | id [u8; 16] | crc32 u32 | bytes
//! footer   entries: (start_ns u64 | end_ns u64 | seq u32 | offset u64 | len u32 | cut u8 | id [u8; 16])* | count u32 | crc32 u32 | "CNKX"
//! ```
//!
//! `id` is the payload the record absorbed (its Tier 0 id), so a chunk a killed worker left open can be committed
//! from the file alone ([`crate::recover`]); a record written without one (a test fixture) has the
//! nil id. Version 1 files (no id in the record or the footer entry) are still read, with the nil id for every
//! record; the writer writes version 2.
//!
//! A frame's capture provenance goes in a provenance record just before the frame's own record: the same
//! head with tag "P", the frame's payload id, zero times and sequence, and 14 bytes (`layout u8` (1) |
//! `device_seq u64` | `epoch u32` | `flags u8`). It is not a frame: the footer never lists it and frame indexes
//! don't count it, so readers that go by the footer never see it. Only crash recovery reads it
//! ([`read_provenance`]), so a recovered chunk keeps what capture attested. The tag is additive and the file
//! version stays 2: a file without provenance records reads as before, and an older reader's scan steps over one as
//! an unreadable stretch. A purge rewrite drops them, since a rewritten chunk's provenance is in the catalog already.
//!
//! Opus audio uses the same container: about a second of Opus packets per record, laid out by [`crate::gop`]
//! ([`ChunkFormat::Opus`]), and so does a raw camera's H.264 ([`ChunkFormat::H264Au`]). One crash-safe container and
//! one footer index for everything means a purge cuts whole records as tombstones. A playable file comes from the
//! export command.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use continuum_catalog::{ChunkFrame, FrameProvenance};
use continuum_core::{StreamId, Uuid};

const MAGIC: &[u8; 4] = b"CNK1";
const END: &[u8; 4] = b"CNKX";
const VERSION: u16 = 2;
const HEADER_LEN: u64 = 24;
/// The record head and footer entry the writer writes (version 2: with the payload id).
const RECORD_HEAD: u64 = 45;
const FOOTER_ENTRY: usize = 8 + 8 + 4 + 8 + 4 + 1 + 16;
const NIL: Uuid = Uuid([0; 16]);

/// The sizes a file's version gives its records and footer entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Layout {
    head: u64,
    entry: usize,
    /// Records and entries carry the payload id (version 2).
    ids: bool,
}

const V1: Layout = Layout { head: 29, entry: 8 + 8 + 4 + 8 + 4 + 1, ids: false };
const V2: Layout = Layout { head: RECORD_HEAD, entry: FOOTER_ENTRY, ids: true };

impl Layout {
    fn of(version: u16) -> Option<Self> {
        match version {
            1 => Some(V1),
            2 => Some(V2),
            _ => None,
        }
    }
}
const TAG_LIVE: u8 = b'R';
const TAG_CUT: u8 = b'T';
const TAG_PROVENANCE: u8 = b'P';
/// The layout byte a provenance record's bytes start with.
const PROVENANCE_LAYOUT: u8 = 1;
const PROVENANCE_LEN: usize = 1 + 8 + 4 + 1;

/// Why a chunk couldn't be written or read.
#[derive(Debug, thiserror::Error)]
pub enum ChunkError {
    /// The file isn't a chunk, or its header is from another version.
    #[error("{0}: not a chunk file")]
    NotAChunk(PathBuf),
    /// A record's CRC didn't match its bytes.
    #[error("{path}: record {index} is corrupt")]
    Corrupt {
        /// The chunk.
        path: PathBuf,
        /// The record's index.
        index: usize,
    },
    /// A frame index past the end of the chunk.
    #[error("frame {0} is not in the chunk")]
    NoSuchFrame(usize),
    /// A payload larger than a record can hold.
    #[error("payload of {0} bytes is too large for a record")]
    TooLarge(usize),
    /// The file system failed.
    #[error(transparent)]
    Io(#[from] io::Error),
}

/// What kind of bytes a chunk's records hold, so a reader knows how to interpret them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ChunkFormat {
    /// The producer's bytes as written: JPEG frames from an MJPEG camera, for instance.
    Passthrough = 1,
    /// Raw frames compressed with LZ4 ([`crate::codecs::lz4_compress`]).
    Lz4 = 2,
    /// PCM audio encoded as FLAC ([`crate::codecs::flac_encode`]).
    Flac = 3,
    /// Arrow IPC bytes, compressed by Arrow itself.
    ArrowIpc = 4,
    /// Tier 2 audio: about a second of Opus packets per record, laid out by [`crate::gop`].
    Opus = 6,
    /// Microphone ring batches, one per record, as FLAC or LZ4 by [`crate::pcm`].
    PcmBatch = 7,
    /// An ordinary MP4 registered in place ([`crate::inplace_video`]): never written by the tier worker,
    /// and its frames are GOPs, read through [`crate::inplace_video`], not [`ChunkReader`].
    Mp4 = 8,
    /// Tier 1 video of a raw camera: one H.264 access unit per record (one record per Tier 0 frame), as a
    /// one-unit [`crate::gop`] record carrying the unit's times and keyframe flag. The GOP is a second; a chunk starts
    /// on a keyframe, so it decodes on its own, and a frame decodes from the keyframe before it in its chunk.
    H264Au = 10,
    /// [`ChunkFormat::H264Au`] re-encoded to AV1 by the opt-in archive pass ([`crate::av1`]): the same records
    /// (payload ids, times, sequence numbers, tombstones) with one AV1 temporal unit each, in the same one-unit
    /// [`crate::gop`] record. The first unit is a keyframe carrying the sequence header, so the chunk decodes on its
    /// own; units are in presentation order (AV1 has no reordered output units).
    Av1Au = 11,
}

impl ChunkFormat {
    fn from_tag(tag: u32) -> Option<Self> {
        match tag {
            1 => Some(Self::Passthrough),
            2 => Some(Self::Lz4),
            3 => Some(Self::Flac),
            4 => Some(Self::ArrowIpc),
            6 => Some(Self::Opus),
            7 => Some(Self::PcmBatch),
            8 => Some(Self::Mp4),
            10 => Some(Self::H264Au),
            11 => Some(Self::Av1Au),
            _ => None,
        }
    }
}

/// One record's place and time: the catalog's [`ChunkFrame`] plus what the catalog doesn't keep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Record {
    /// The record's start time, UTC nanoseconds.
    pub start_ns: u64,
    /// Its end time; equal to `start_ns` for an instant.
    pub end_ns: u64,
    /// Its sequence number within the stream.
    pub seq: u32,
    /// The first byte of its encoded extent in the file.
    pub offset: u64,
    /// The encoded extent's length.
    pub len: u32,
    /// A tombstone left by a purge: the bytes are gone, the place is kept so frame indexes never move.
    pub cut: bool,
    /// The payload this record absorbed (nil for a record written without one, and for every record of a version
    /// 1 file).
    pub id: Uuid,
}

impl Record {
    /// The catalog's view of this record.
    #[must_use]
    pub fn frame(&self) -> ChunkFrame {
        ChunkFrame { start_ns: self.start_ns, offset: self.offset, length: u64::from(self.len), cut: self.cut }
    }

    /// The last nanosecond this record covers, exclusive: `end_ns`, or `start_ns + 1` for an instant.
    #[must_use]
    pub fn end_exclusive(&self) -> u64 {
        if self.end_ns > self.start_ns { self.end_ns } else { self.start_ns.saturating_add(1) }
    }
}

/// The CRC over a record head's fields (everything after the tag and before the CRC itself) and its bytes.
fn record_crc(head: &[u8], bytes: &[u8]) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(&head[1..head.len() - 4]);
    h.update(bytes);
    h.finalize()
}

fn head_crc(head: &[u8]) -> u32 {
    u32::from_le_bytes(head[head.len() - 4..].try_into().expect("4 bytes"))
}

fn record_head(
    tag: u8,
    id: Uuid,
    len: u32,
    start_ns: u64,
    end_ns: u64,
    seq: u32,
    bytes: &[u8],
) -> [u8; RECORD_HEAD as usize] {
    let mut head = [0u8; RECORD_HEAD as usize];
    head[0] = tag;
    head[1..5].copy_from_slice(&len.to_le_bytes());
    head[5..13].copy_from_slice(&start_ns.to_le_bytes());
    head[13..21].copy_from_slice(&end_ns.to_le_bytes());
    head[21..25].copy_from_slice(&seq.to_le_bytes());
    head[25..41].copy_from_slice(&id.0);
    let crc = record_crc(&head, bytes);
    head[41..45].copy_from_slice(&crc.to_le_bytes());
    head
}

/// A record from its head (already checked) at `pos`.
fn record_of(head: &[u8], pos: u64, layout: Layout) -> Record {
    let mut id = NIL;
    if layout.ids {
        id.0.copy_from_slice(&head[25..41]);
    }
    Record {
        start_ns: u64::from_le_bytes(head[5..13].try_into().expect("8")),
        end_ns: u64::from_le_bytes(head[13..21].try_into().expect("8")),
        seq: u32::from_le_bytes(head[21..25].try_into().expect("4")),
        offset: pos + layout.head,
        len: u32::from_le_bytes(head[1..5].try_into().expect("4")),
        cut: head[0] == TAG_CUT,
        id,
    }
}

/// Writes one chunk, record by record, and closes it with a footer.
pub struct ChunkWriter {
    path: PathBuf,
    file: File,
    pos: u64,
    records: Vec<Record>,
    stream: StreamId,
    format: ChunkFormat,
    created_ns: u64,
}

impl ChunkWriter {
    /// Creates `path` (refusing to overwrite) and writes the header.
    ///
    /// # Errors
    ///
    /// An I/O error, including the file existing already.
    pub fn create(path: &Path, stream: StreamId, format: ChunkFormat, created_ns: u64) -> Result<Self, ChunkError> {
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        let mut header = Vec::with_capacity(HEADER_LEN as usize);
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&VERSION.to_le_bytes());
        header.extend_from_slice(&0u16.to_le_bytes());
        header.extend_from_slice(&stream.0.to_le_bytes());
        header.extend_from_slice(&(format as u32).to_le_bytes());
        header.extend_from_slice(&created_ns.to_le_bytes());
        file.write_all(&header)?;
        Ok(Self { path: path.to_path_buf(), file, pos: HEADER_LEN, records: Vec::new(), stream, format, created_ns })
    }

    /// Sets the native camera orientation before the first record. Legacy chunks have no flip flag.
    /// # Errors
    /// An I/O error, or an attempt to change orientation after records have been appended.
    pub fn set_flip(&mut self, flip: bool) -> Result<(), ChunkError> {
        if !self.records.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "set chunk orientation before appending").into());
        }
        self.file.seek(SeekFrom::Start(6))?;
        self.file.write_all(&u16::from(flip).to_le_bytes())?;
        self.file.seek(SeekFrom::Start(self.pos))?;
        Ok(())
    }

    /// The chunk's path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The stream this chunk holds.
    #[must_use]
    pub fn stream(&self) -> StreamId {
        self.stream
    }

    /// What its records hold.
    #[must_use]
    pub fn format(&self) -> ChunkFormat {
        self.format
    }

    /// When the chunk was created, UTC nanoseconds.
    #[must_use]
    pub fn created_ns(&self) -> u64 {
        self.created_ns
    }

    /// Records written so far, in order.
    #[must_use]
    pub fn records(&self) -> &[Record] {
        &self.records
    }

    /// The time span of the records so far: the first start to the last exclusive end.
    #[must_use]
    pub fn span(&self) -> Option<(u64, u64)> {
        let first = self.records.first()?;
        let end = self.records.iter().map(Record::end_exclusive).max()?;
        Some((first.start_ns, end))
    }

    /// Appends one record that absorbs no Tier 0 payload (a Tier 2 GOP, a fixture): the nil id. See
    /// [`ChunkWriter::append_payload`].
    ///
    /// # Errors
    ///
    /// [`ChunkError::TooLarge`] past `u32::MAX` bytes, or an I/O error.
    pub fn append(&mut self, start_ns: u64, end_ns: u64, seq: u32, bytes: &[u8]) -> Result<Record, ChunkError> {
        self.write_record(TAG_LIVE, NIL, start_ns, end_ns, seq, bytes)
    }

    /// Appends one payload with its id. The bytes are in the file when this returns (a killed worker loses none of
    /// them); [`ChunkWriter::sync`] or [`ChunkWriter::close`] makes them durable against a power loss.
    ///
    /// # Errors
    ///
    /// [`ChunkError::TooLarge`] past `u32::MAX` bytes, or an I/O error.
    pub fn append_payload(
        &mut self,
        id: Uuid,
        start_ns: u64,
        end_ns: u64,
        seq: u32,
        bytes: &[u8],
    ) -> Result<Record, ChunkError> {
        self.write_record(TAG_LIVE, id, start_ns, end_ns, seq, bytes)
    }

    /// Writes what capture attested about payload `id`, ahead of the payload's own record (see the module docs). It
    /// is not a frame: [`ChunkWriter::records`] and the footer don't list it.
    ///
    /// # Errors
    ///
    /// An I/O error.
    pub fn append_provenance(&mut self, id: Uuid, provenance: FrameProvenance) -> Result<(), ChunkError> {
        let mut bytes = [0u8; PROVENANCE_LEN];
        bytes[0] = PROVENANCE_LAYOUT;
        bytes[1..9].copy_from_slice(&provenance.device_seq.to_le_bytes());
        bytes[9..13].copy_from_slice(&provenance.epoch.to_le_bytes());
        bytes[13] = provenance.flags();
        let head = record_head(TAG_PROVENANCE, id, PROVENANCE_LEN as u32, 0, 0, 0, &bytes);
        self.file.write_all(&head)?;
        self.file.write_all(&bytes)?;
        self.pos += RECORD_HEAD + PROVENANCE_LEN as u64;
        Ok(())
    }

    fn write_record(
        &mut self,
        tag: u8,
        id: Uuid,
        start_ns: u64,
        end_ns: u64,
        seq: u32,
        bytes: &[u8],
    ) -> Result<Record, ChunkError> {
        let len = u32::try_from(bytes.len()).map_err(|_| ChunkError::TooLarge(bytes.len()))?;
        let head = record_head(tag, id, len, start_ns, end_ns, seq, bytes);
        self.file.write_all(&head)?;
        self.file.write_all(bytes)?;
        let record = Record { start_ns, end_ns, seq, offset: self.pos + RECORD_HEAD, len, cut: tag == TAG_CUT, id };
        self.pos += RECORD_HEAD + u64::from(len);
        self.records.push(record);
        Ok(record)
    }

    /// Asks the file system to make everything appended so far durable.
    ///
    /// # Errors
    ///
    /// An I/O error.
    pub fn sync(&mut self) -> Result<(), ChunkError> {
        self.file.sync_data()?;
        Ok(())
    }

    /// Writes the footer, makes the file durable, and returns the records.
    ///
    /// # Errors
    ///
    /// An I/O error.
    pub fn close(mut self) -> Result<Vec<Record>, ChunkError> {
        write_footer(&mut self.file, &self.records)?;
        self.file.sync_all()?;
        Ok(self.records)
    }
}

fn write_footer<W: Write>(w: &mut W, records: &[Record]) -> Result<(), ChunkError> {
    let mut body = Vec::with_capacity(records.len() * FOOTER_ENTRY + 4);
    for r in records {
        body.extend_from_slice(&r.start_ns.to_le_bytes());
        body.extend_from_slice(&r.end_ns.to_le_bytes());
        body.extend_from_slice(&r.seq.to_le_bytes());
        body.extend_from_slice(&r.offset.to_le_bytes());
        body.extend_from_slice(&r.len.to_le_bytes());
        body.push(u8::from(r.cut));
        body.extend_from_slice(&r.id.0);
    }
    let count = u32::try_from(records.len()).map_err(|_| ChunkError::TooLarge(records.len()))?;
    body.extend_from_slice(&count.to_le_bytes());
    let crc = crc32fast::hash(&body);
    w.write_all(&body)?;
    w.write_all(&crc.to_le_bytes())?;
    w.write_all(END)?;
    Ok(())
}

/// A chunk opened for reading, with its index.
pub struct ChunkReader {
    path: PathBuf,
    file: File,
    stream: StreamId,
    format: ChunkFormat,
    created_ns: u64,
    records: Vec<Record>,
    flip: bool,
    footer_ok: bool,
    skipped: usize,
    layout: Layout,
}

impl ChunkReader {
    /// Whether native camera pixels need a 180-degree rotation for display. False for legacy chunks.
    #[must_use]
    pub fn flip(&self) -> bool {
        self.flip
    }

    /// Opens a chunk. With a valid footer the index comes from it; without one (the writer died, or the chunk was
    /// still open) the records are recovered by scanning. A record whose CRC fails is skipped and the scan resyncs
    /// on the next record that checks out ([`ChunkReader::skipped`] counts them); a short record at the end is the
    /// one being written when the writer died.
    ///
    /// # Errors
    ///
    /// [`ChunkError::NotAChunk`] for a file without the header, or an I/O error.
    pub fn open(path: &Path) -> Result<Self, ChunkError> {
        Self::open_selected(path, None)?.ok_or_else(|| ChunkError::NotAChunk(path.to_path_buf()))
    }

    /// Open only if the container index overlaps the requested window, including tombstones.
    ///
    /// # Errors
    /// The same index/header errors as [`Self::open`], and payload errors for an overlapping archive.
    pub fn open_in(path: &Path, start_ns: u64, end_ns: u64) -> Result<Option<Self>, ChunkError> {
        if start_ns >= end_ns {
            return Ok(None);
        }
        Self::open_selected(path, Some((start_ns, end_ns)))
    }

    fn open_selected(path: &Path, window: Option<(u64, u64)>) -> Result<Option<Self>, ChunkError> {
        let mut file = File::open(path)?;
        let mut header = [0u8; HEADER_LEN as usize];
        if file.read_exact(&mut header).is_err() {
            return Err(ChunkError::NotAChunk(path.to_path_buf()));
        }
        if &header[..4] != MAGIC {
            return Err(ChunkError::NotAChunk(path.to_path_buf()));
        }
        let version = u16::from_le_bytes([header[4], header[5]]);
        let stream = StreamId(u32::from_le_bytes([header[8], header[9], header[10], header[11]]));
        let tag = u32::from_le_bytes([header[12], header[13], header[14], header[15]]);
        let created_ns = u64::from_le_bytes(header[16..24].try_into().expect("8 bytes"));
        let Some(format) = ChunkFormat::from_tag(tag) else { return Err(ChunkError::NotAChunk(path.to_path_buf())) };
        let Some(layout) = Layout::of(version) else {
            return Err(ChunkError::NotAChunk(path.to_path_buf()));
        };
        let len = file.metadata()?.len();
        let (records, footer_ok, skipped) = match read_footer(&mut file, len, layout)? {
            Some(records) => (records, true, 0),
            None => {
                let (records, skipped) = scan_records(&mut file, len, layout)?;
                (records, false, skipped)
            }
        };
        if let Some((start_ns, end_ns)) = window
            && !records
                .iter()
                .map(|r| r.start_ns)
                .min()
                .zip(records.iter().map(Record::end_exclusive).max())
                .is_some_and(|(first, last)| first < end_ns && last > start_ns)
        {
            return Ok(None);
        }
        Ok(Some(Self {
            path: path.to_path_buf(),
            file,
            stream,
            format,
            created_ns,
            records,
            flip: u16::from_le_bytes([header[6], header[7]]) & 1 != 0,
            footer_ok,
            skipped,
            layout,
        }))
    }
    /// The chunk's path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The stream this chunk holds.
    #[must_use]
    pub fn stream(&self) -> StreamId {
        self.stream
    }

    /// What its records hold.
    #[must_use]
    pub fn format(&self) -> ChunkFormat {
        self.format
    }

    /// When the chunk was created, UTC nanoseconds.
    #[must_use]
    pub fn created_ns(&self) -> u64 {
        self.created_ns
    }

    /// The index, in record order, tombstones included.
    #[must_use]
    pub fn records(&self) -> &[Record] {
        &self.records
    }

    /// Whether the index came from a valid footer (false means it was recovered by scanning).
    #[must_use]
    pub fn finalized(&self) -> bool {
        self.footer_ok
    }

    /// Corrupt stretches the recovery scan skipped (always 0 with a footer).
    #[must_use]
    pub fn skipped(&self) -> usize {
        self.skipped
    }

    /// The live records whose time overlaps `[start_ns, end_ns)`, with their indexes.
    #[must_use]
    pub fn records_in(&self, start_ns: u64, end_ns: u64) -> Vec<(usize, Record)> {
        self.records
            .iter()
            .enumerate()
            .filter(|(_, r)| !r.cut && r.start_ns < end_ns && r.end_exclusive() > start_ns)
            .map(|(i, r)| (i, *r))
            .collect()
    }

    /// Reads one record's payload, checking its CRC. A tombstone reads as empty.
    ///
    /// # Errors
    ///
    /// [`ChunkError::NoSuchFrame`], [`ChunkError::Corrupt`], or an I/O error.
    pub fn read(&mut self, index: usize) -> Result<Vec<u8>, ChunkError> {
        let r = *self.records.get(index).ok_or(ChunkError::NoSuchFrame(index))?;
        if r.cut {
            return Ok(Vec::new());
        }
        let corrupt = || ChunkError::Corrupt { path: self.path.clone(), index };
        let layout = self.layout;
        if r.offset < HEADER_LEN + layout.head {
            return Err(corrupt());
        }
        let mut head = vec![0u8; layout.head as usize];
        self.file.seek(SeekFrom::Start(r.offset - layout.head))?;
        self.file.read_exact(&mut head)?;
        let len = u32::from_le_bytes(head[1..5].try_into().expect("4 bytes"));
        if head[0] != TAG_LIVE || len != r.len {
            return Err(corrupt());
        }
        let crc = head_crc(&head);
        let mut bytes = vec![0u8; r.len as usize];
        self.file.read_exact(&mut bytes)?;
        if record_crc(&head, &bytes) != crc {
            return Err(corrupt());
        }
        Ok(bytes)
    }
}

fn read_footer(file: &mut File, len: u64, layout: Layout) -> Result<Option<Vec<Record>>, ChunkError> {
    if len < HEADER_LEN + 12 {
        return Ok(None);
    }
    let mut tail = [0u8; 12];
    file.seek(SeekFrom::Start(len - 12))?;
    file.read_exact(&mut tail)?;
    if &tail[8..12] != END {
        return Ok(None);
    }
    let count = u32::from_le_bytes(tail[0..4].try_into().expect("4 bytes")) as usize;
    let crc = u32::from_le_bytes(tail[4..8].try_into().expect("4 bytes"));
    let body_len = count * layout.entry + 4;
    if (body_len as u64) + 8 > len - HEADER_LEN {
        return Ok(None);
    }
    let start = len - 8 - body_len as u64;
    let mut body = vec![0u8; body_len];
    file.seek(SeekFrom::Start(start))?;
    file.read_exact(&mut body)?;
    if crc32fast::hash(&body) != crc {
        return Ok(None);
    }
    let mut records = Vec::with_capacity(count);
    for e in body[..count * layout.entry].chunks_exact(layout.entry) {
        let mut id = NIL;
        if layout.ids {
            id.0.copy_from_slice(&e[33..49]);
        }
        records.push(Record {
            start_ns: u64::from_le_bytes(e[0..8].try_into().expect("8")),
            end_ns: u64::from_le_bytes(e[8..16].try_into().expect("8")),
            seq: u32::from_le_bytes(e[16..20].try_into().expect("4")),
            offset: u64::from_le_bytes(e[20..28].try_into().expect("8")),
            len: u32::from_le_bytes(e[28..32].try_into().expect("4")),
            cut: e[32] != 0,
            id,
        });
    }
    Ok(Some(records))
}

/// A record [`parse_at`] found, with a provenance record's bytes (`None` for a frame).
type Parsed = (Record, Option<Vec<u8>>);

/// A valid frame record or provenance record at `pos`, with the provenance record's bytes (`None` for a frame).
/// `Ok(None)` means no valid record starts there (a short tail, or a CRC that doesn't check out); `Err` is a real
/// I/O error.
fn parse_at(file: &mut File, len: u64, pos: u64, layout: Layout) -> Result<Option<Parsed>, ChunkError> {
    if pos + layout.head > len {
        return Ok(None);
    }
    let mut head = vec![0u8; layout.head as usize];
    file.seek(SeekFrom::Start(pos))?;
    file.read_exact(&mut head)?;
    let provenance = head[0] == TAG_PROVENANCE && layout.ids;
    if head[0] != TAG_LIVE && head[0] != TAG_CUT && !provenance {
        return Ok(None);
    }
    let rlen = u32::from_le_bytes(head[1..5].try_into().expect("4"));
    let data_end = pos + layout.head + u64::from(rlen);
    if data_end > len {
        return Ok(None);
    }
    let crc = head_crc(&head);
    let mut bytes = vec![0u8; rlen as usize];
    file.read_exact(&mut bytes)?;
    if record_crc(&head, &bytes) != crc {
        return Ok(None);
    }
    Ok(Some((record_of(&head, pos, layout), provenance.then_some(bytes))))
}

/// Recovers records by scanning. A record that doesn't parse is skipped by moving one byte at a time until a
/// record parses again (its CRC makes a false match negligible); the count of skipped stretches comes back too.
fn scan_records(file: &mut File, len: u64, layout: Layout) -> Result<(Vec<Record>, usize), ChunkError> {
    let mut records = Vec::new();
    let mut skipped = 0usize;
    let mut pos = HEADER_LEN;
    let mut in_gap = false;
    while pos + layout.head <= len {
        match parse_at(file, len, pos, layout)? {
            Some((r, provenance)) => {
                if in_gap {
                    skipped += 1;
                    in_gap = false;
                }
                pos = r.offset + u64::from(r.len);
                if provenance.is_none() {
                    records.push(r);
                }
            }
            None => {
                in_gap = true;
                pos += 1;
            }
        }
    }
    Ok((records, skipped))
}

/// The records of a chunk being written, from file offset `from` on, for a reader that follows a growing chunk without
/// rescanning it whole each time (the gateway's stream listing re-read every live chunk from the start, about 1.8 s
/// every few seconds). `from` is the offset the previous call returned, or 0 for the start. Returns the records
/// found (frames only, in file order, as [`ChunkReader::open`]'s scan keeps them) and the offset to continue from:
/// just past the last whole record, so one half-written at the end is read whole next time. Once the chunk has its
/// footer it's finished: the footer's records from `from` on come back with no offset, as there's nothing more to
/// follow.
///
/// # Errors
///
/// [`ChunkError::NotAChunk`] for a file without the header, or an I/O error.
pub fn scan_tail(path: &Path, from: u64) -> Result<(Vec<Record>, Option<u64>), ChunkError> {
    let mut file = File::open(path)?;
    let mut header = [0u8; HEADER_LEN as usize];
    if file.read_exact(&mut header).is_err() || &header[..4] != MAGIC {
        return Err(ChunkError::NotAChunk(path.to_path_buf()));
    }
    let Some(layout) = Layout::of(u16::from_le_bytes([header[4], header[5]])) else {
        return Err(ChunkError::NotAChunk(path.to_path_buf()));
    };
    let len = file.metadata()?.len();
    if let Some(all) = read_footer(&mut file, len, layout)? {
        return Ok((all.into_iter().filter(|r| r.offset >= from).collect(), None));
    }
    let mut pos = from.max(HEADER_LEN);
    let mut resume = pos;
    let mut records = Vec::new();
    while pos + layout.head <= len {
        match parse_at(&mut file, len, pos, layout)? {
            Some((r, provenance)) => {
                pos = r.offset + u64::from(r.len);
                resume = pos;
                if provenance.is_none() {
                    records.push(r);
                }
            }
            None => pos += 1,
        }
    }
    Ok((records, Some(resume)))
}

/// What the chunk's provenance records say, by payload id (see the module docs): empty for a chunk written without
/// them, or a version 1 file. A provenance record with a layout or flag this version doesn't
/// know is left out, so its frame stays unknown.
///
/// # Errors
///
/// [`ChunkError::NotAChunk`] or an I/O error.
pub fn read_provenance(path: &Path) -> Result<BTreeMap<Uuid, FrameProvenance>, ChunkError> {
    let mut file = File::open(path)?;
    let mut header = [0u8; HEADER_LEN as usize];
    if file.read_exact(&mut header).is_err() || &header[..4] != MAGIC {
        return Err(ChunkError::NotAChunk(path.to_path_buf()));
    }
    let Some(layout) = Layout::of(u16::from_le_bytes([header[4], header[5]])) else {
        return Err(ChunkError::NotAChunk(path.to_path_buf()));
    };
    let mut out = BTreeMap::new();
    if !layout.ids {
        return Ok(out);
    }
    let len = file.metadata()?.len();
    let mut pos = HEADER_LEN;
    while pos + layout.head <= len {
        match parse_at(&mut file, len, pos, layout)? {
            Some((r, bytes)) => {
                pos = r.offset + u64::from(r.len);
                if let Some(b) = bytes
                    && b.len() == PROVENANCE_LEN
                    && b[0] == PROVENANCE_LAYOUT
                    && let Ok(p) = FrameProvenance::from_wire(
                        u64::from_le_bytes(b[1..9].try_into().expect("8")),
                        u32::from_le_bytes(b[9..13].try_into().expect("4")),
                        b[13],
                    )
                {
                    out.insert(r.id, p);
                }
            }
            None => pos += 1,
        }
    }
    Ok(out)
}

/// Finalizes a chunk left without a footer (the writer died): drops the trailing bytes after the last whole
/// record (a record that was being written) and writes the footer over them. Returns the recovered records. A
/// chunk that already has a footer is left alone. Corrupt records inside the chunk were skipped by the scan and are
/// not in the index; their bytes stay in the file, unreachable.
///
/// # Errors
///
/// [`ChunkError::NotAChunk`] or an I/O error.
pub fn finalize(path: &Path) -> Result<Vec<Record>, ChunkError> {
    let reader = ChunkReader::open(path)?;
    if reader.finalized() {
        return Ok(reader.records().to_vec());
    }
    let records = reader.records().to_vec();
    drop(reader);
    let end = records.last().map_or(HEADER_LEN, |r| r.offset + u64::from(r.len));
    let mut file = OpenOptions::new().write(true).open(path)?;
    file.set_len(end)?;
    file.seek(SeekFrom::Start(end))?;
    write_footer(&mut file, &records)?;
    file.sync_all()?;
    Ok(records)
}

/// Moves a chunk's records from index `keep` on (in file order) into a new chunk file at `to`, with the same stream,
/// format, and flip, and each record's id, times, sequence, and provenance record; records a purge cut are left
/// behind. The new file is finished and durable before the old one is cut back to its first `keep` records and
/// given a footer, so a crash in between leaves the moved records in both files, never in neither. Returns the new
/// file's records; an empty list (and no new file) when there is nothing past `keep` to move.
///
/// For a part the catalog refused for good (docs/CONTINUUM-STORAGE.md): the chunk keeps what the catalog
/// holds of it, and the rest becomes a chunk of its own.
///
/// # Errors
///
/// [`ChunkError::NotAChunk`], [`ChunkError::Corrupt`] for a record that doesn't read back, or an I/O error.
pub fn split_off(path: &Path, keep: usize, to: &Path, created_ns: u64) -> Result<Vec<Record>, ChunkError> {
    let mut reader = ChunkReader::open(path)?;
    let all = reader.records().to_vec();
    if keep >= all.len() || all[keep..].iter().all(|r| r.cut) {
        return Ok(Vec::new());
    }
    let provenance = read_provenance(path)?;
    let mut writer = ChunkWriter::create(to, reader.stream(), reader.format(), created_ns)?;
    writer.set_flip(reader.flip())?;
    for (index, r) in all.iter().enumerate().skip(keep).filter(|(_, r)| !r.cut) {
        let bytes = reader.read(index)?;
        if let Some(attested) = provenance.get(&r.id) {
            writer.append_provenance(r.id, *attested)?;
        }
        writer.append_payload(r.id, r.start_ns, r.end_ns, r.seq, &bytes)?;
    }
    let moved = writer.close()?;
    if let Some(dir) = to.parent() {
        File::open(dir)?.sync_all()?;
    }
    drop(reader);
    let end = keep.checked_sub(1).map_or(HEADER_LEN, |last| all[last].offset + u64::from(all[last].len));
    let mut file = OpenOptions::new().write(true).open(path)?;
    file.set_len(end)?;
    file.seek(SeekFrom::Start(end))?;
    write_footer(&mut file, &all[..keep])?;
    file.sync_all()?;
    Ok(moved)
}

/// Deletes a chunk file. Missing is fine (a purge may run twice).
///
/// # Errors
///
/// An I/O error other than not found.
pub fn remove(path: &Path) -> Result<(), ChunkError> {
    match fs::remove_file(path) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    crate::layout::prune_hour_dir(path); // an hour aged out whole leaves no directory behind
    Ok(())
}

/// What [`rewrite_without`] produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rewritten {
    /// Every record, in its original index position; cut ones are tombstones.
    pub records: Vec<Record>,
    /// The new file's length, for the catalog's locator.
    pub length: u64,
}

/// `path` with `.partial` added: where a file is written before it's renamed into place, so a failure never leaves
/// a half-written file under the real name. Recovery deletes leftovers in the chunk directory.
pub(crate) fn partial(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".partial");
    name.into()
}

/// Rewrites a chunk with the records at `cut` (by index) turned into tombstones, into a new file beside it that
/// replaces it by rename. Every record keeps its index, so the catalog's frame numbers stay valid; offsets change,
/// and the catalog's `rewrite_chunk` takes the new ones.
///
/// # Errors
///
/// [`ChunkError::NoSuchFrame`] for a cut index past the end, or a read or write error.
pub fn rewrite_without(path: &Path, cut: &[usize]) -> Result<Rewritten, ChunkError> {
    let mut reader = ChunkReader::open(path)?;
    let records = reader.records().to_vec();
    if let Some(bad) = cut.iter().find(|i| **i >= records.len()) {
        return Err(ChunkError::NoSuchFrame(*bad));
    }
    let tmp = path.with_extension("rewrite");
    let _ = fs::remove_file(&tmp);
    let mut writer = ChunkWriter::create(&tmp, reader.stream(), reader.format(), reader.created_ns())?;
    writer.set_flip(reader.flip())?;
    for (i, r) in records.iter().enumerate() {
        if r.cut || cut.contains(&i) {
            writer.write_record(TAG_CUT, r.id, r.start_ns, r.end_ns, r.seq, &[])?;
        } else {
            let bytes = reader.read(i)?;
            writer.append_payload(r.id, r.start_ns, r.end_ns, r.seq, &bytes)?;
        }
    }
    let out = writer.close()?;
    drop(reader);
    let length = fs::metadata(&tmp)?.len();
    fs::rename(&tmp, path)?;
    if let Some(dir) = path.parent() {
        File::open(dir)?.sync_all()?;
    }
    Ok(Rewritten { records: out, length })
}

/// A chunk file's format, from its header alone (the AV1 archive pass's scan reads thousands of these).
///
/// # Errors
///
/// [`ChunkError::NotAChunk`] for a file that isn't one, or an I/O error.
pub fn format_of(path: &Path) -> Result<ChunkFormat, ChunkError> {
    let mut header = [0u8; HEADER_LEN as usize];
    File::open(path)?.read_exact(&mut header).map_err(|_| ChunkError::NotAChunk(path.to_path_buf()))?;
    let tag = u32::from_le_bytes([header[12], header[13], header[14], header[15]]);
    match ChunkFormat::from_tag(tag) {
        Some(format) if &header[..4] == MAGIC => Ok(format),
        _ => Err(ChunkError::NotAChunk(path.to_path_buf())),
    }
}

/// Rewrites a chunk as `format`, with each live record's bytes replaced by `bytes[i]` (by index), into a new file
/// beside it that replaces it by rename, as [`rewrite_without`] does: every record keeps its index, payload id, times,
/// and sequence number, and tombstones stay tombstones. The caller checks the file is still the one it read (the
/// rename replaces whatever is there).
///
/// # Errors
///
/// [`ChunkError::NoSuchFrame`] when `bytes` doesn't have one entry per record, or holds none for a live record (or
/// some for a tombstone); a read or write error.
pub fn rewrite_as(path: &Path, format: ChunkFormat, bytes: &[Option<Vec<u8>>]) -> Result<Rewritten, ChunkError> {
    let reader = ChunkReader::open(path)?;
    let records = reader.records().to_vec();
    if bytes.len() != records.len() {
        return Err(ChunkError::NoSuchFrame(bytes.len().min(records.len())));
    }
    if let Some(bad) = records.iter().zip(bytes).position(|(r, b)| r.cut == b.is_some()) {
        return Err(ChunkError::NoSuchFrame(bad));
    }
    let tmp = path.with_extension("rewrite");
    let _ = fs::remove_file(&tmp);
    let mut writer = ChunkWriter::create(&tmp, reader.stream(), format, reader.created_ns())?;
    writer.set_flip(reader.flip())?;
    for (r, b) in records.iter().zip(bytes) {
        match b {
            Some(b) => writer.append_payload(r.id, r.start_ns, r.end_ns, r.seq, b)?,
            None => writer.write_record(TAG_CUT, r.id, r.start_ns, r.end_ns, r.seq, &[])?,
        };
    }
    let out = writer.close()?;
    drop(reader);
    let length = fs::metadata(&tmp)?.len();
    fs::rename(&tmp, path)?;
    if let Some(dir) = path.parent() {
        File::open(dir)?.sync_all()?;
    }
    Ok(Rewritten { records: out, length })
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn frames(n: usize) -> Vec<Vec<u8>> {
        (0..n).map(|i| vec![u8::try_from(i % 251).expect("small"); 100 + i]).collect()
    }

    fn write_chunk(path: &Path, data: &[Vec<u8>]) -> Vec<Record> {
        let mut w = ChunkWriter::create(path, StreamId(7), ChunkFormat::Passthrough, 5).expect("create");
        for (i, f) in data.iter().enumerate() {
            let t = 1_000 + (i as u64) * 33;
            w.append(t, t, i as u32, f).expect("append");
        }
        w.close().expect("close")
    }

    #[test]
    fn camera_flip_survives_recovery_and_purge_rewriting() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("camera.cnk");
        let mut writer = ChunkWriter::create(&path, StreamId(1), ChunkFormat::Passthrough, 0).unwrap();
        writer.set_flip(true).unwrap();
        writer.append(1, 1, 0, b"first").unwrap();
        assert!(writer.set_flip(false).is_err());
        writer.append(2, 2, 1, b"second").unwrap();
        writer.sync().unwrap();
        drop(writer);
        assert!(ChunkReader::open(&path).unwrap().flip());
        finalize(&path).unwrap();
        assert!(ChunkReader::open(&path).unwrap().flip());
        rewrite_without(&path, &[0]).unwrap();
        let mut reader = ChunkReader::open(&path).unwrap();
        assert!(reader.flip());
        assert_eq!(reader.read(1).unwrap(), b"second");
    }

    #[test]
    fn records_round_trip_through_the_footer() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("a.cnk");
        let data = frames(5);
        let written = write_chunk(&path, &data);
        let mut r = ChunkReader::open(&path).expect("open");
        assert!(r.finalized());
        assert_eq!((r.stream(), r.format(), r.created_ns()), (StreamId(7), ChunkFormat::Passthrough, 5));
        assert_eq!(r.records(), written.as_slice());
        for (i, f) in data.iter().enumerate() {
            assert_eq!(r.read(i).expect("read"), *f);
        }
        assert_eq!(r.records_in(1_033, 1_067).iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(r.records_in(1_000, 1_001).len(), 1, "an instant at the slice start");
        assert!(r.records_in(1_132, 1_132).is_empty(), "an empty slice");
        assert!(matches!(r.read(9), Err(ChunkError::NoSuchFrame(9))));
    }

    #[test]
    fn a_killed_writer_loses_nothing_it_appended() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("k.cnk");
        let mut w = ChunkWriter::create(&path, StreamId(1), ChunkFormat::Passthrough, 0).expect("create");
        let data = frames(4);
        for (i, f) in data.iter().enumerate() {
            w.append(i as u64, i as u64 + 1, i as u32, f).expect("append");
        }
        // SIGKILL: no drop, no sync, no close.
        std::mem::forget(w);
        let mut r = ChunkReader::open(&path).expect("open");
        assert!(!r.finalized());
        assert_eq!(r.records().len(), 4);
        assert_eq!(r.read(3).expect("read"), data[3]);
    }

    fn id(n: u8) -> Uuid {
        let mut id = [0u8; 16];
        id[6] = 0x70;
        id[15] = n;
        Uuid(id)
    }

    #[test]
    fn records_carry_their_payload_ids_through_the_footer_and_a_scan() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("ids.cnk");
        let data = frames(3);
        let mut w = ChunkWriter::create(&path, StreamId(1), ChunkFormat::Passthrough, 0).expect("create");
        for (i, f) in data.iter().enumerate() {
            let r = w.append_payload(id(i as u8 + 1), i as u64, i as u64 + 1, i as u32, f).expect("append");
            assert_eq!(r.id, id(i as u8 + 1));
        }
        let r = w.append(9, 9, 9, b"no payload").expect("append");
        assert_eq!(r.id, NIL, "a record written without a payload has the nil id");
        w.close().expect("close");
        let r = ChunkReader::open(&path).expect("open");
        assert!(r.finalized());
        assert_eq!(r.records().iter().map(|r| r.id).collect::<Vec<_>>(), [id(1), id(2), id(3), NIL], "from the footer");

        // Without a footer the ids come from the records themselves.
        let scan = dir.path().join("scan.cnk");
        let mut w = ChunkWriter::create(&scan, StreamId(1), ChunkFormat::Passthrough, 0).expect("create");
        w.append_payload(id(7), 1, 2, 0, &data[0]).expect("append");
        std::mem::forget(w);
        let r = ChunkReader::open(&scan).expect("open");
        assert!(!r.finalized());
        assert_eq!(r.records()[0].id, id(7), "from the record head");
        let recovered = finalize(&scan).expect("finalize");
        assert_eq!(recovered[0].id, id(7));
        assert_eq!(ChunkReader::open(&scan).expect("open").records()[0].id, id(7), "and the footer written for it");
    }

    /// A version 1 file written by hand (29-byte record heads, 33-byte footer entries, no ids).
    fn write_v1(path: &Path, data: &[Vec<u8>]) {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&7u32.to_le_bytes());
        out.extend_from_slice(&(ChunkFormat::Passthrough as u32).to_le_bytes());
        out.extend_from_slice(&5u64.to_le_bytes());
        let mut entries = Vec::new();
        for (i, f) in data.iter().enumerate() {
            let t = 1_000 + (i as u64) * 33;
            let mut head = [0u8; 29];
            head[0] = TAG_LIVE;
            head[1..5].copy_from_slice(&(f.len() as u32).to_le_bytes());
            head[5..13].copy_from_slice(&t.to_le_bytes());
            head[13..21].copy_from_slice(&t.to_le_bytes());
            head[21..25].copy_from_slice(&(i as u32).to_le_bytes());
            let crc = record_crc(&head, f);
            head[25..29].copy_from_slice(&crc.to_le_bytes());
            let offset = out.len() as u64 + 29;
            out.extend_from_slice(&head);
            out.extend_from_slice(f);
            entries.extend_from_slice(&t.to_le_bytes());
            entries.extend_from_slice(&t.to_le_bytes());
            entries.extend_from_slice(&(i as u32).to_le_bytes());
            entries.extend_from_slice(&offset.to_le_bytes());
            entries.extend_from_slice(&(f.len() as u32).to_le_bytes());
            entries.push(0);
        }
        entries.extend_from_slice(&(data.len() as u32).to_le_bytes());
        let crc = crc32fast::hash(&entries);
        out.extend_from_slice(&entries);
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(END);
        fs::write(path, out).expect("write");
    }

    #[test]
    fn a_version_1_chunk_is_read_with_nil_ids_from_its_footer_and_from_a_scan() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("v1.cnk");
        let data = frames(3);
        write_v1(&path, &data);
        let mut r = ChunkReader::open(&path).expect("open");
        assert!(r.finalized());
        assert_eq!(r.records().len(), 3);
        assert!(r.records().iter().all(|r| r.id == NIL));
        assert_eq!(r.records()[1].start_ns, 1_033);
        for (i, f) in data.iter().enumerate() {
            assert_eq!(r.read(i).expect("read"), *f);
        }
        // Cut the footer off: the records are scanned with the version 1 layout.
        let len = fs::metadata(&path).expect("meta").len();
        let body = fs::read(&path).expect("read");
        fs::write(&path, &body[..(len as usize) - (3 * 33 + 12)]).expect("truncate");
        let mut r = ChunkReader::open(&path).expect("open");
        assert!(!r.finalized());
        assert_eq!(r.records().len(), 3);
        assert_eq!(r.read(2).expect("read"), data[2]);
        // Rewriting carries the records into the current version, ids and all.
        rewrite_without(&path, &[1]).expect("rewrite");
        let r = ChunkReader::open(&path).expect("open");
        assert!(r.records()[1].cut && r.records().iter().all(|r| r.id == NIL));
    }

    #[test]
    fn a_chunks_tail_moves_to_a_new_file_and_the_chunk_keeps_its_head() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (path, to) = (dir.path().join("7-5-1.cnk"), dir.path().join("7-9-2.cnk"));
        let mut w = ChunkWriter::create(&path, StreamId(7), ChunkFormat::H264Au, 5).expect("create");
        w.set_flip(true).expect("flip");
        let data = frames(5);
        let attested = FrameProvenance { device_seq: 40, epoch: 3, stamped_at_arrival: true };
        for (i, f) in data.iter().enumerate() {
            if i == 3 {
                w.append_provenance(id(i as u8), attested).expect("provenance");
            }
            w.append_payload(id(i as u8), i as u64 * 10, i as u64 * 10 + 1, i as u32, f).expect("append");
        }
        w.sync().expect("sync"); // still open: no footer

        let moved = split_off(&path, 3, &to, 9).expect("split");
        assert_eq!(
            moved.iter().map(|r| (r.id, r.start_ns, r.seq)).collect::<Vec<_>>(),
            [(id(3), 30, 3), (id(4), 40, 4)]
        );
        let mut new = ChunkReader::open(&to).expect("the new chunk");
        assert!(new.finalized() && new.flip());
        assert_eq!((new.stream(), new.format(), new.created_ns()), (StreamId(7), ChunkFormat::H264Au, 9));
        assert_eq!(new.records(), moved.as_slice());
        assert_eq!((new.read(0).expect("read"), new.read(1).expect("read")), (data[3].clone(), data[4].clone()));
        assert_eq!(read_provenance(&to).expect("provenance"), BTreeMap::from([(id(3), attested)]));

        let mut old = ChunkReader::open(&path).expect("the old chunk");
        assert!(old.finalized(), "it has its footer now");
        assert_eq!(old.records().iter().map(|r| r.id).collect::<Vec<_>>(), [id(0), id(1), id(2)]);
        assert_eq!(old.read(2).expect("read"), data[2]);
        assert!(read_provenance(&path).expect("provenance").is_empty(), "the moved frame's record went with it");

        let nothing = dir.path().join("7-9-3.cnk");
        assert!(split_off(&path, 3, &nothing, 9).expect("nothing past it").is_empty());
        assert!(!nothing.exists());
    }

    #[test]
    fn a_chunk_without_a_footer_is_recovered_and_finalized() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("b.cnk");
        let mut w = ChunkWriter::create(&path, StreamId(1), ChunkFormat::Lz4, 0).expect("create");
        let data = frames(4);
        for (i, f) in data.iter().enumerate() {
            w.append(i as u64, i as u64 + 1, i as u32, f).expect("append");
        }
        w.sync().expect("sync");
        drop(w);
        let mut f = OpenOptions::new().append(true).open(&path).expect("append");
        f.write_all(b"R\x10\x00\x00\x00partial").expect("partial");
        drop(f);

        let mut r = ChunkReader::open(&path).expect("open");
        assert!(!r.finalized());
        assert_eq!(r.records().len(), 4);
        assert_eq!(r.skipped(), 0);
        assert_eq!(r.read(3).expect("read"), data[3]);
        drop(r);

        let recovered = finalize(&path).expect("finalize");
        assert_eq!(recovered.len(), 4);
        let r = ChunkReader::open(&path).expect("open");
        assert!(r.finalized());
        assert_eq!(r.records(), recovered.as_slice());
        assert_eq!(finalize(&path).expect("idempotent").len(), 4);
    }

    #[test]
    fn a_growing_chunk_is_followed_by_its_tail_alone() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("g.cnk");
        let mut w = ChunkWriter::create(&path, StreamId(1), ChunkFormat::Lz4, 0).expect("create");
        let data = frames(7);
        for (i, f) in data.iter().take(4).enumerate() {
            w.append(i as u64, i as u64 + 1, i as u32, f).expect("append");
        }
        w.sync().expect("sync");
        let (first, at) = scan_tail(&path, 0).expect("scan");
        let at = at.expect("still being written");
        assert_eq!(first.len(), 4);
        assert_eq!(scan_tail(&path, at).expect("again"), (vec![], Some(at))); // nothing new: nothing read
        for (i, f) in data.iter().enumerate().skip(4) {
            w.append(i as u64, i as u64 + 1, i as u32, f).expect("append");
        }
        w.sync().expect("sync");
        let (rest, end) = scan_tail(&path, at).expect("tail");
        let end = end.expect("still being written");
        assert_eq!(rest.iter().map(|r| r.seq).collect::<Vec<_>>(), vec![4, 5, 6]);
        assert!(end > at);
        w.close().expect("close");
        assert_eq!(scan_tail(&path, end).expect("finished"), (vec![], None), "its footer: nothing more to follow");
        let (whole, done) = scan_tail(&path, 0).expect("whole");
        assert_eq!((whole.len(), done), (7, None));
        let path = dir.path().join("h.cnk");
        let mut w = ChunkWriter::create(&path, StreamId(1), ChunkFormat::Lz4, 0).expect("create");
        for (i, f) in data.iter().enumerate() {
            w.append(i as u64, i as u64 + 1, i as u32, f).expect("append");
        }
        w.sync().expect("sync");
        let (first, _) = scan_tail(&path, 0).expect("scan");
        let end = first.last().map(|r| r.offset + u64::from(r.len)).expect("records");
        drop(w);
        let mut f = OpenOptions::new().append(true).open(&path).expect("append");
        f.write_all(b"R\x10\x00\x00\x00partial").expect("partial"); // a record half-written at the end
        drop(f);
        assert_eq!(scan_tail(&path, end).expect("partial"), (vec![], Some(end))); // read whole next time
        assert_eq!(first, ChunkReader::open(&path).expect("open").records());
    }

    #[test]
    fn provenance_records_survive_a_crash_without_becoming_frames() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("p.cnk");
        let id = |n: u8| Uuid([n; 16]);
        let attested = |n: u64| FrameProvenance { device_seq: 500 + n, epoch: 77, stamped_at_arrival: n == 2 };
        let mut w = ChunkWriter::create(&path, StreamId(1), ChunkFormat::Passthrough, 0).expect("create");
        w.append_provenance(id(0), attested(0)).expect("provenance");
        w.append_payload(id(0), 10, 10, 0, b"zero").expect("append");
        // Frame 1's producer said nothing about it.
        w.append_payload(id(1), 20, 20, 1, b"one").expect("append");
        w.append_provenance(id(2), attested(2)).expect("provenance");
        w.append_payload(id(2), 30, 30, 2, b"two").expect("append");
        // A layout this version doesn't know leaves its frame unknown rather than misread.
        let mut future = [0u8; PROVENANCE_LEN];
        future[0] = PROVENANCE_LAYOUT + 1;
        let head = record_head(TAG_PROVENANCE, id(3), PROVENANCE_LEN as u32, 0, 0, 0, &future);
        w.file.write_all(&head).expect("head");
        w.file.write_all(&future).expect("bytes");
        w.pos += RECORD_HEAD + PROVENANCE_LEN as u64;
        w.append_payload(id(3), 40, 40, 3, b"three").expect("append");
        // The worker dies between a frame's provenance record and the frame itself.
        w.append_provenance(id(4), attested(4)).expect("provenance");
        w.sync().expect("sync");
        drop(w);

        let mut r = ChunkReader::open(&path).expect("open");
        assert!(!r.finalized());
        assert_eq!(r.records().iter().map(|r| r.id).collect::<Vec<_>>(), [id(0), id(1), id(2), id(3)]);
        assert_eq!(r.skipped(), 0, "a provenance record is not a corrupt stretch");
        assert_eq!(r.read(2).expect("read"), b"two");
        drop(r);
        let records = finalize(&path).expect("finalize");
        assert_eq!(records.len(), 4);
        assert_eq!(ChunkReader::open(&path).expect("open").records(), records.as_slice(), "the footer agrees");
        let got = read_provenance(&path).expect("provenance");
        assert_eq!(got, BTreeMap::from([(id(0), attested(0)), (id(2), attested(2))]), "the orphan went with the tail");

        // A chunk written without provenance records reads as before.
        let old = dir.path().join("old.cnk");
        let recs = write_chunk(&old, &frames(3));
        assert!(read_provenance(&old).expect("provenance").is_empty());
        assert_eq!(ChunkReader::open(&old).expect("open").records(), recs.as_slice());
    }

    #[test]
    fn a_corrupt_record_is_refused_and_the_scan_resyncs_past_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("c.cnk");
        let data = frames(3);
        let recs = write_chunk(&path, &data);
        let mut f = OpenOptions::new().write(true).open(&path).expect("open");
        f.seek(SeekFrom::Start(recs[1].offset + 3)).expect("seek");
        f.write_all(&[0xff]).expect("write");
        drop(f);
        let mut r = ChunkReader::open(&path).expect("open");
        assert!(matches!(r.read(1), Err(ChunkError::Corrupt { index: 1, .. })));
        assert_eq!(r.read(2).expect("others fine"), data[2]);
        drop(r);
        // Without the footer, the scan skips the corrupt record and resyncs on the next one.
        let len = fs::metadata(&path).expect("meta").len();
        OpenOptions::new().write(true).open(&path).expect("open").set_len(len - 12).expect("truncate");
        let mut r = ChunkReader::open(&path).expect("open");
        assert_eq!(r.records().len(), 2);
        assert_eq!(r.skipped(), 1);
        assert_eq!(r.read(1).expect("read"), data[2]);
        drop(r);
        // A flipped header time is caught too (the CRC covers the header fields).
        let path2 = dir.path().join("c2.cnk");
        let recs = write_chunk(&path2, &data);
        let mut f = OpenOptions::new().write(true).open(&path2).expect("open");
        f.seek(SeekFrom::Start(recs[0].offset - RECORD_HEAD + 6)).expect("seek");
        f.write_all(&[0x7f]).expect("write");
        drop(f);
        let mut r = ChunkReader::open(&path2).expect("open");
        assert!(matches!(r.read(0), Err(ChunkError::Corrupt { index: 0, .. })));
    }

    #[test]
    fn rewriting_without_a_record_leaves_a_tombstone_in_its_place() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("d.cnk");
        let data = frames(4);
        write_chunk(&path, &data);
        assert!(matches!(rewrite_without(&path, &[9]), Err(ChunkError::NoSuchFrame(9))));
        let out = rewrite_without(&path, &[1]).expect("rewrite");
        assert_eq!(out.records.len(), 4);
        assert!(out.records[1].cut && out.records[1].len == 0);
        assert_eq!(out.length, fs::metadata(&path).expect("meta").len());
        let mut r = ChunkReader::open(&path).expect("open");
        assert_eq!(r.records(), out.records.as_slice());
        assert_eq!(r.created_ns(), 5, "the creation time is kept");
        assert_eq!(r.read(0).expect("read"), data[0]);
        assert_eq!(r.read(1).expect("tombstone"), Vec::<u8>::new());
        assert_eq!(r.read(2).expect("read"), data[2]);
        assert_eq!(r.records_in(0, u64::MAX).len(), 3, "a tombstone is not in a slice");
        assert!(!dir.path().join("d.rewrite").exists());
        let bytes = fs::read(&path).expect("read");
        assert!(!bytes.windows(data[1].len()).any(|w| w == data[1].as_slice()), "the cut bytes are gone");
    }

    #[test]
    fn not_a_chunk_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("e.cnk");
        fs::write(&path, b"hello").expect("write");
        assert!(matches!(ChunkReader::open(&path), Err(ChunkError::NotAChunk(_))));
        assert!(remove(&dir.path().join("missing.cnk")).is_ok());
    }

    fn payloads() -> impl Strategy<Value = Vec<Vec<u8>>> {
        prop::collection::vec(prop::collection::vec(any::<u8>(), 0..300), 0..12)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn any_records_round_trip_with_and_without_a_footer(data in payloads()) {
            let dir = tempfile::tempdir().expect("tempdir");
            let path = dir.path().join("p.cnk");
            let written = write_chunk(&path, &data);
            let mut r = ChunkReader::open(&path).expect("open");
            prop_assert_eq!(r.records(), written.as_slice());
            for (i, f) in data.iter().enumerate() {
                prop_assert_eq!(r.read(i).expect("read"), f.clone());
            }
            drop(r);
            let len = fs::metadata(&path).expect("meta").len();
            let footer = (data.len() * FOOTER_ENTRY + 4 + 8) as u64;
            OpenOptions::new().write(true).open(&path).expect("open").set_len(len - footer).expect("truncate");
            let r = ChunkReader::open(&path).expect("open");
            prop_assert!(!r.finalized());
            prop_assert_eq!(r.records(), written.as_slice());
        }

        #[test]
        fn a_chunk_followed_in_two_scans_has_the_records_of_one(data in payloads(), cut_fraction in 0.0f64..1.0) {
            let dir = tempfile::tempdir().expect("tempdir");
            let path = dir.path().join("f.cnk");
            let written = write_chunk(&path, &data);
            let len = fs::metadata(&path).expect("meta").len();
            let footer = (data.len() * FOOTER_ENTRY + 4 + 8) as u64;
            OpenOptions::new().write(true).open(&path).expect("open").set_len(len - footer).expect("unfinish");
            let whole = fs::read(&path).expect("read");
            let cut = HEADER_LEN + (((whole.len() as u64 - HEADER_LEN) as f64) * cut_fraction) as u64;
            OpenOptions::new().write(true).open(&path).expect("open").set_len(cut).expect("truncate");
            let (head, at) = scan_tail(&path, 0).expect("head");
            fs::write(&path, &whole).expect("grow");
            let (tail, _) = scan_tail(&path, at.expect("no footer yet")).expect("tail");
            prop_assert_eq!([head, tail].concat(), written);
        }

        #[test]
        fn a_truncated_chunk_recovers_a_prefix_of_its_records(data in payloads(), cut_fraction in 0.0f64..1.0) {
            let dir = tempfile::tempdir().expect("tempdir");
            let path = dir.path().join("t.cnk");
            let written = write_chunk(&path, &data);
            let len = fs::metadata(&path).expect("meta").len();
            let keep = HEADER_LEN + (((len - HEADER_LEN) as f64) * cut_fraction) as u64;
            OpenOptions::new().write(true).open(&path).expect("open").set_len(keep).expect("truncate");
            let r = ChunkReader::open(&path).expect("open");
            let n = r.records().len();
            prop_assert!(n <= written.len());
            prop_assert_eq!(r.records(), &written[..n]);
            prop_assert_eq!(r.skipped(), 0);
            drop(r);
            let recovered = finalize(&path).expect("finalize");
            prop_assert_eq!(recovered.len(), n);
            prop_assert!(ChunkReader::open(&path).expect("open").finalized());
        }

        #[test]
        fn any_cut_set_rewrites_to_tombstones_and_keeps_the_rest(
            data in payloads(), cuts in prop::collection::vec(0usize..12, 0..4)
        ) {
            let dir = tempfile::tempdir().expect("tempdir");
            let path = dir.path().join("r.cnk");
            write_chunk(&path, &data);
            let valid: Vec<usize> = cuts.iter().copied().filter(|c| *c < data.len()).collect();
            let out = rewrite_without(&path, &valid).expect("rewrite");
            prop_assert_eq!(out.records.len(), data.len());
            let mut r = ChunkReader::open(&path).expect("open");
            for (i, f) in data.iter().enumerate() {
                let got = r.read(i).expect("read");
                if valid.contains(&i) {
                    prop_assert!(r.records()[i].cut);
                    prop_assert!(got.is_empty());
                } else {
                    prop_assert_eq!(got, f.clone());
                }
            }
            prop_assert!(!path.with_extension("rewrite").exists());
        }
    }
}
