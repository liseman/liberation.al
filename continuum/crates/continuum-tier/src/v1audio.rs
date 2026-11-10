//! v1's audio archive registered as Tier 1 chunks in place, for the speech importer (#548).
//!
//! v1 recorded each microphone as FLAC files of about five minutes (`audio/<device>/<day>/<HHMMSS>.flac`), with each
//! file's exact start in the device's `index.jsonl` (`{"file", "start" (UTC seconds), "sr"}`). Each file becomes one
//! payload and one chunk on its own stream (`<device>.audio_v1`, shape `PCM_S16LEx<channels>`), without being moved,
//! copied, or re-encoded:
//!
//! 1. The file's STREAMINFO gives its rate, channels, depth, and sample count, so its span is
//!    `[start, start + samples / rate)`. A file with no samples (v1 died while writing it) is skipped with a reason.
//! 2. One header-only payload is ingested for the file, then the file is committed as a chunk of one frame (offset
//!    0, the whole file) absorbing that payload. Its uri is the file's path and its format FLAC.
//! 3. Both ids are deterministic ([`v1_ids`]): a UUIDv7 whose time is the file's start and whose other bits hash
//!    the path, the sample count, and the start. A second run sends identical headers, which the catalog takes as
//!    repeats (the ingest returns the existing sequence, the commit answers `duplicate`).
//!
//! Reading back: [`crate::hydrate`] and [`crate::hydrate_frame`] recognize a raw FLAC file and return it whole as
//! one record ([`crate::ChunkFormat::Flac`]), and [`decode_flac`] gives its samples as f32, scaled as capture's are.
//! A v1 file isn't in the tier worker's chunk directory, so its purges don't rewrite it (see the module's PR).

use std::path::{Path, PathBuf};

use continuum_client::{ChunkFrame, Client, NewChunk, StreamSpec, StreamStorage};
use continuum_core::schema::continuum as fb;
use continuum_core::{DomainLayer, HeaderBuf, HeaderSpec, PayloadFormat, StorageLocator, StorageTier, StreamId, Uuid};

use crate::TierError;

/// What goes wrong registering v1 audio.
#[derive(Debug, thiserror::Error)]
pub enum V1Error {
    /// A file couldn't be read, or isn't FLAC.
    #[error(transparent)]
    Tier(#[from] TierError),
    /// The broker refused, or the connection failed.
    #[error(transparent)]
    Client(#[from] continuum_client::Error),
    /// A line of `index.jsonl` that can't be used.
    #[error("{0}")]
    Index(String),
    /// The catalog holds something else for the file's time: the ingest's refusal and the commit's, together.
    #[error("{path}: the catalog refused it (ingest: {ingest}; commit: {commit})")]
    Refused {
        /// The file.
        path: PathBuf,
        /// Why the payload's ingest was refused.
        ingest: String,
        /// Why the chunk's commit was refused.
        commit: String,
    },
}

/// A FLAC file's STREAMINFO.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlacInfo {
    /// Frames per second.
    pub rate_hz: u32,
    /// Interleaved channels.
    pub channels: u16,
    /// Bits per sample.
    pub bits: u32,
    /// Frames (samples per channel); 0 when the writer never finished the header.
    pub frames: u64,
}

impl FlacInfo {
    /// The file's length in nanoseconds.
    #[must_use]
    pub fn duration_ns(&self) -> u64 {
        if self.rate_hz == 0 {
            return 0;
        }
        u64::try_from(u128::from(self.frames) * 1_000_000_000 / u128::from(self.rate_hz)).unwrap_or(u64::MAX)
    }

    /// The stream shape for this layout: `PCM_S16LEx<channels>` for 16 bit, `PCM_S24LE3x<channels>` for 24,
    /// `PCM_S32LEx<channels>` for 32, and `PCM_S<bits>x<channels>` for any other depth.
    #[must_use]
    pub fn shape_name(&self) -> String {
        let kind = match self.bits {
            16 => "S16LE".to_owned(),
            24 => "S24LE3".to_owned(),
            32 => "S32LE".to_owned(),
            b => format!("S{b}"),
        };
        format!("PCM_{kind}x{}", self.channels)
    }
}

/// Reads a FLAC file's STREAMINFO (only its first block).
///
/// # Errors
///
/// [`TierError::Codec`] for a file that isn't FLAC, or an I/O error.
pub fn flac_info(path: &Path) -> Result<FlacInfo, TierError> {
    let reader = claxon::FlacReader::open(path).map_err(|e| TierError::Codec(format!("{}: {e}", path.display())))?;
    let s = reader.streaminfo();
    Ok(FlacInfo {
        rate_hz: s.sample_rate,
        channels: u16::try_from(s.channels).unwrap_or(0),
        bits: s.bits_per_sample,
        frames: s.samples.unwrap_or(0),
    })
}

/// Whether `bytes` start like a FLAC stream.
#[must_use]
pub fn is_flac(bytes: &[u8]) -> bool {
    bytes.starts_with(b"fLaC")
}

/// Decodes FLAC bytes to interleaved f32 samples, each the integer divided by 2^(bits - 1), as capture's f32 is.
/// Returns the rate, the channel count, and the samples.
///
/// # Errors
///
/// [`TierError::Codec`] for bytes that aren't FLAC, or that are damaged.
pub fn decode_flac(bytes: &[u8]) -> Result<(u32, u16, Vec<f32>), TierError> {
    let bad = |e: claxon::Error| TierError::Codec(format!("flac: {e}"));
    let mut reader = claxon::FlacReader::new(bytes).map_err(bad)?;
    let info = reader.streaminfo();
    if !(1..=32).contains(&info.bits_per_sample) {
        return Err(TierError::Codec(format!("flac: {} bits per sample", info.bits_per_sample)));
    }
    #[expect(clippy::cast_precision_loss, reason = "a power of two, exact")]
    let scale = (1u64 << (info.bits_per_sample - 1)) as f32;
    let mut out = Vec::new();
    for s in reader.samples() {
        #[expect(clippy::cast_precision_loss, reason = "at most 24 significant bits in practice, as capture's are")]
        out.push(s.map_err(bad)? as f32 / scale);
    }
    Ok((info.sample_rate, u16::try_from(info.channels).unwrap_or(0), out))
}

/// The payload's and the chunk's ids for a v1 file: UUIDv7s whose 48 bit time is the file's start in milliseconds
/// and whose other bits are an FNV-1a hash of the path, the frame count, and the start (a different seed for each),
/// so the same file always gets the same ids.
#[must_use]
pub fn v1_ids(path: &Path, frames: u64, start_ns: u64) -> (Uuid, Uuid) {
    ids_seeded("audio", path, frames, start_ns)
}

/// [`v1_ids`] with `kind` (`audio`, `video`) in the seeds, so each kind of v1 file has its own ids.
pub(crate) fn ids_seeded(kind: &str, path: &Path, frames: u64, start_ns: u64) -> (Uuid, Uuid) {
    let key = |seed: &[u8]| {
        let mut bytes = seed.to_vec();
        bytes.extend_from_slice(path.as_os_str().as_encoded_bytes());
        bytes.push(0);
        bytes.extend_from_slice(&frames.to_le_bytes());
        bytes.extend_from_slice(&start_ns.to_le_bytes());
        v7(start_ns, fnv1a(&bytes, 0xcbf2_9ce4_8422_2325), fnv1a(&bytes, 0x8422_2325_cbf2_9ce4))
    };
    let payload = format!("continuum v1 {kind} payload\0");
    let chunk = format!("continuum v1 {kind} chunk\0");
    (key(payload.as_bytes()), key(chunk.as_bytes()))
}

/// 64-bit FNV-1a from `basis`: stable across builds and platforms, unlike `std`'s hasher.
fn fnv1a(bytes: &[u8], basis: u64) -> u64 {
    bytes.iter().fold(basis, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3))
}

fn v7(start_ns: u64, a: u64, b: u64) -> Uuid {
    let ms = start_ns / 1_000_000;
    let mut u = [0u8; 16];
    u[..6].copy_from_slice(&ms.to_be_bytes()[2..]);
    u[6..14].copy_from_slice(&a.to_be_bytes());
    u[14..].copy_from_slice(&b.to_be_bytes()[..2]);
    u[6] = 0x70 | (u[6] & 0x0f);
    u[8] = 0x80 | (u[8] & 0x3f);
    Uuid(u)
}

/// What registering one v1 file did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum V1Registered {
    /// The file is a chunk now (or was already: `duplicate`).
    Registered {
        /// Its payload.
        payload_id: Uuid,
        /// Its chunk.
        chunk_id: Uuid,
        /// Its span, UTC nanoseconds, half-open.
        span: (u64, u64),
        /// It was registered by an earlier run.
        duplicate: bool,
    },
    /// It was passed over, and why.
    Skipped(String),
}

/// The stream a v1 device's archive goes on: `<name>`, a sensor at layer 1, its files' FLAC as raw bytes. Nothing
/// is ever written to its Tier 0 storage; the smallest slab class is named because one has to be.
#[must_use]
pub fn v1_stream_spec(name: &str, info: &FlacInfo) -> StreamSpec {
    StreamSpec {
        name: name.to_owned(),
        kind: fb::StreamKind::Sensor,
        format: fb::PayloadFormat::RawBytes,
        shape_name: info.shape_name(),
        shape_version: 1,
        nominal_rate_hz: 0.0,
        storage: StreamStorage::Slab { class: 0 },
        anchor: None,
        layer: fb::DomainLayer::Layer1Raw,
    }
}

/// Registers one v1 FLAC file, which starts at `start_ns`, as a payload and a Tier 1 chunk on `stream` (registered
/// by this session with [`v1_stream_spec`]). The session must have registered a worker (`register_worker`), since
/// only a worker commits chunks. Idempotent: see the module doc. The file's canonical path (symlinks and `..`
/// resolved) is its uri and part of its ids, so a run from another directory or through a symlink is still a repeat.
/// A file that isn't readable FLAC, or has no samples, is skipped with a reason.
///
/// # Errors
///
/// [`V1Error::Tier`] for an I/O error, [`V1Error::Client`] when the broker refuses, and [`V1Error::Refused`] when
/// the catalog holds something else at the file's time.
pub async fn register_v1_flac(
    client: &Client,
    stream: StreamId,
    path: &Path,
    start_ns: u64,
) -> Result<V1Registered, V1Error> {
    let path = std::fs::canonicalize(path).map_err(TierError::from)?;
    let info = match flac_info(&path) {
        Ok(info) => info,
        Err(TierError::Codec(why)) => return Ok(V1Registered::Skipped(format!("not readable FLAC: {why}"))),
        Err(e) => return Err(e.into()),
    };
    if info.frames == 0 || info.rate_hz == 0 {
        return Ok(V1Registered::Skipped(format!(
            "{}: no samples in its STREAMINFO (v1 stopped while writing it)",
            path.display()
        )));
    }
    let end_ns = start_ns.saturating_add(info.duration_ns()).max(start_ns.saturating_add(1));
    let (payload_id, chunk_id) = v1_ids(&path, info.frames, start_ns);
    // Read before ingesting, so a file that vanished doesn't leave a header-only payload behind.
    let length = std::fs::metadata(&path).map_err(TierError::from)?.len();
    let shape = InPlace {
        stream,
        span: (start_ns, end_ns),
        format: PayloadFormat::RawBytes,
        shape_name: info.shape_name(),
        frames: vec![ChunkFrame { start_ns, offset: 0, length, cut: false }],
        payloads: vec![Absorbed { id: payload_id, span: (start_ns, end_ns), seq: 0 }],
    };
    register_in_place(client, &path, &shape, chunk_id, length, "v1 audio").await
}

/// A v1 file to register in place: its stream, span, payload format and shape, its chunk's frame index, and the
/// payloads the chunk absorbs, one per frame.
pub(crate) struct InPlace {
    pub(crate) stream: StreamId,
    pub(crate) span: (u64, u64),
    pub(crate) format: PayloadFormat,
    pub(crate) shape_name: String,
    pub(crate) frames: Vec<ChunkFrame>,
    pub(crate) payloads: Vec<Absorbed>,
}

/// A payload a chunk registered in place absorbs: its deterministic id, span, and sequence number.
pub(crate) struct Absorbed {
    pub(crate) id: Uuid,
    pub(crate) span: (u64, u64),
    pub(crate) seq: u32,
}

/// Ingests the header-only payloads for the file at `path` (canonical) and commits the file in place as a Tier 1
/// chunk (`chunk_id`) absorbing them. Idempotent: see the module doc.
pub(crate) async fn register_in_place(
    client: &Client,
    path: &Path,
    f: &InPlace,
    chunk_id: Uuid,
    length: u64,
    what: &str,
) -> Result<V1Registered, V1Error> {
    let (start_ns, end_ns) = f.span;
    let header = |id: Uuid, span: (u64, u64), seq: u32, chunk: bool| -> Result<HeaderBuf, TierError> {
        let mut spec = HeaderSpec::raw(id, f.stream, span.0, span.1);
        spec.seq = seq;
        spec.layer = DomainLayer::Layer1Raw;
        spec.format = f.format;
        spec.shape_name.clone_from(&f.shape_name);
        spec.shape_version = 1;
        if chunk {
            spec.source_tier = StorageTier::Tier1Warm;
        }
        Ok(HeaderBuf::build(&spec)?)
    };
    // Registered already? Then nothing is ingested: a payload the chunk absorbed has no row any more, so ingesting
    // its header again would make a new, orphaned one.
    if let Some(done) = committed_already(client, f, chunk_id, path).await? {
        return Ok(done);
    }
    // A repeat of a header the catalog has since absorbed into the chunk is refused as a Conflict (the chunk's header
    // holds the payload's time key) rather than taken as a repeat (#685), so a Conflict here is settled by the
    // commit: a duplicate means an earlier run registered the file.
    let mut conflict = None;
    for p in &f.payloads {
        match client.ingest_acked(&header(p.id, p.span, p.seq, false)?, None).await {
            Ok(_) => {}
            Err(e) if e.broker_code() == Some(fb::ErrorCode::Conflict) => conflict = conflict.or(Some(e)),
            Err(e) => return Err(e.into()),
        }
    }
    let absorbed: Vec<Uuid> = f.payloads.iter().map(|p| p.id).collect();
    let chunk = NewChunk {
        header: header(chunk_id, f.span, 0, true)?,
        location: StorageLocator::Chunk {
            tier: StorageTier::Tier1Warm,
            uri: path.to_string_lossy().into_owned(),
            offset: 0,
            length,
        },
        frames: f.frames.clone(),
    };
    let done = match (client.commit_chunk(&chunk, &absorbed).await, conflict) {
        (Ok(done), None) => {
            if done.duplicate {
                // The payload went in fresh but the chunk was there: only possible if the catalog stops refusing the
                // repeat (#685) by some other route than taking it as one. The header-only payload is then an orphan.
                eprintln!("{what}: {}: ingested again though its chunk was committed already", path.display());
            }
            done
        }
        (Ok(done), Some(e)) => {
            if !done.duplicate {
                // Committed fresh after a refused ingest: a purge in flight over the span made the frame a cut one.
                eprintln!("{what}: {}: committed with its frames cut ({e})", path.display());
            }
            done
        }
        (Err(e), None) => return Err(e.into()),
        (Err(commit), Some(ingest)) => {
            return Err(V1Error::Refused {
                path: path.to_path_buf(),
                ingest: ingest.to_string(),
                commit: commit.to_string(),
            });
        }
    };
    Ok(V1Registered::Registered {
        payload_id: absorbed.first().copied().unwrap_or(Uuid([0; 16])),
        chunk_id: done.chunk_id,
        span: (start_ns, end_ns),
        duplicate: done.duplicate,
    })
}

/// The registration of a file whose chunk (`chunk_id`, at `path`) the catalog already has, or `None`.
async fn committed_already(
    client: &Client,
    f: &InPlace,
    chunk_id: Uuid,
    path: &Path,
) -> Result<Option<V1Registered>, V1Error> {
    let streams = client.list_streams().await?;
    let Some(name) = streams.iter().find(|s| s.id == f.stream).map(|s| s.name.clone()) else { return Ok(None) };
    let uri = path.to_string_lossy();
    let found = client.query_all(&continuum_client::QuerySpec::hit_only(name, f.span)).await?;
    let known = found.items.iter().any(|i| {
        i.header.id() == chunk_id && matches!(&i.locator, Some(StorageLocator::Chunk { uri: u, .. }) if *u == uri)
    });
    Ok(known.then(|| V1Registered::Registered {
        payload_id: f.payloads.first().map_or(Uuid([0; 16]), |p| p.id),
        chunk_id,
        span: f.span,
        duplicate: true,
    }))
}

/// UTC seconds as a float (v1's `start`) to nanoseconds: the whole seconds and the fraction converted apart, since
/// `start * 1e9` rounds to a multiple of 256 ns at today's times. 1793300000.25 is 1793300000250000000. `None` for
/// a negative, infinite, or NaN time, or one past what nanoseconds since 1970 hold in a u64 (the year 2554).
#[must_use]
pub fn seconds_to_ns(seconds: f64) -> Option<u64> {
    if !(0.0..18_446_744_073.0).contains(&seconds) {
        return None;
    }
    let whole = seconds.trunc();
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "in range, checked above")]
    let (s, frac) = (whole as u64, ((seconds - whole) * 1e9).round() as u64);
    s.checked_mul(1_000_000_000)?.checked_add(frac)
}

/// One line of v1's `index.jsonl`.
#[derive(serde::Deserialize)]
struct IndexLine {
    file: String,
    start: f64,
}

/// Registers every file a v1 device directory's `index.jsonl` lists, on stream `stream_name` (registered here from
/// the first readable file's layout). A file missing from disk, not readable FLAC, with no samples, or with a
/// different layout from the stream's is skipped with a reason; a file listed twice (v1 restarted within the second
/// and reopened the same name) is registered once, at its last start. A line that isn't JSON is an error, since the
/// index is the archive's only record of its starts.
///
/// # Errors
///
/// [`V1Error::Index`] for an unreadable index or line, and as [`register_v1_flac`].
pub async fn register_v1_index(
    client: &Client,
    stream_name: &str,
    device_dir: &Path,
) -> Result<Vec<(PathBuf, V1Registered)>, V1Error> {
    let index_path = device_dir.join("index.jsonl");
    let text =
        std::fs::read_to_string(&index_path).map_err(|e| V1Error::Index(format!("{}: {e}", index_path.display())))?;
    // In the order files first appear, each at its last start.
    let mut lines: Vec<(PathBuf, u64)> = Vec::new();
    let mut at: std::collections::HashMap<PathBuf, usize> = std::collections::HashMap::new();
    for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let l: IndexLine = serde_json::from_str(line)
            .map_err(|e| V1Error::Index(format!("{} line {}: {e}", index_path.display(), n + 1)))?;
        let start_ns = seconds_to_ns(l.start)
            .ok_or_else(|| V1Error::Index(format!("{} line {}: start {}", index_path.display(), n + 1, l.start)))?;
        let path = device_dir.join(&l.file);
        match at.get(&path) {
            Some(&i) => lines[i].1 = start_ns,
            None => {
                at.insert(path.clone(), lines.len());
                lines.push((path, start_ns));
            }
        }
    }
    let mut stream: Option<(StreamId, String)> = None;
    let mut out = Vec::with_capacity(lines.len());
    for (path, start_ns) in lines {
        if !path.is_file() {
            out.push((path.clone(), V1Registered::Skipped(format!("{}: not on disk", path.display()))));
            continue;
        }
        let info = match flac_info(&path) {
            Ok(info) => info,
            Err(TierError::Codec(why)) => {
                out.push((path, V1Registered::Skipped(format!("not readable FLAC: {why}"))));
                continue;
            }
            Err(e) => return Err(e.into()),
        };
        let (id, shape) = match &stream {
            Some(s) => s.clone(),
            None => {
                let registered = client.register_stream(&v1_stream_spec(stream_name, &info)).await?;
                let s = (StreamId(registered.stream_id), info.shape_name());
                stream = Some(s.clone());
                s
            }
        };
        if info.shape_name() != shape {
            let why = format!("{}: {} on a {shape} stream", path.display(), info.shape_name());
            out.push((path, V1Registered::Skipped(why)));
            continue;
        }
        let r = register_v1_flac(client, id, &path, start_ns).await?;
        out.push((path, r));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_stable_v7s_that_differ_by_file_and_between_payload_and_chunk() {
        let p = Path::new("/data/audio/wave3/2026-10-29/140000.flac");
        let t = 1_793_238_094_991_116_800;
        let (a, b) = v1_ids(p, 14_400_000, t);
        assert_eq!((a, b), v1_ids(p, 14_400_000, t), "the same file, the same ids");
        assert_ne!(a, b);
        assert_ne!(a, v1_ids(p, 14_400_001, t).0);
        assert_ne!(a, v1_ids(Path::new("/data/audio/wave3/2026-10-29/140500.flac"), 14_400_000, t).0);
        let u = uuid::Uuid::from_bytes(a.0);
        assert_eq!(u.get_version_num(), 7);
        assert_eq!(u.get_variant(), uuid::Variant::RFC4122);
        let ms = u.get_timestamp().expect("v7").to_unix();
        assert_eq!(ms.0 * 1_000 + u64::from(ms.1) / 1_000_000, t / 1_000_000);
    }

    #[test]
    fn v1_starts_convert_to_the_nanosecond_they_name() {
        assert_eq!(seconds_to_ns(1_793_300_000.25), Some(1_793_300_000_250_000_000));
        assert_eq!(seconds_to_ns(1_793_238_094.5), Some(1_793_238_094_500_000_000));
        assert_eq!(seconds_to_ns(0.0), Some(0));
        for bad in [-1.0, f64::NAN, f64::INFINITY, 2e10] {
            assert_eq!(seconds_to_ns(bad), None, "{bad}");
        }
    }

    #[test]
    fn a_flac_file_gives_its_info_and_decodes_to_capture_scaled_f32() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("a.flac");
        let samples: Vec<i32> = (0..96_000).map(|i| (i % 2_000) - 1_000).collect();
        std::fs::write(&path, crate::codecs::flac_encode(&samples, 1, 16, 48_000).expect("encode")).expect("write");
        let info = flac_info(&path).expect("info");
        assert_eq!(info, FlacInfo { rate_hz: 48_000, channels: 1, bits: 16, frames: 96_000 });
        assert_eq!(info.duration_ns(), 2_000_000_000);
        assert_eq!(info.shape_name(), "PCM_S16LEx1");
        let (rate, channels, f) = decode_flac(&std::fs::read(&path).expect("read")).expect("decode");
        assert_eq!((rate, channels, f.len()), (48_000, 1, 96_000));
        #[expect(clippy::cast_precision_loss, reason = "test values")]
        let want: Vec<f32> = samples.iter().map(|&s| s as f32 / 32_768.0).collect();
        assert_eq!(f, want);
        assert!(flac_info(&dir.path().join("missing.flac")).is_err());
        assert!(decode_flac(b"not flac").is_err());
    }
}
