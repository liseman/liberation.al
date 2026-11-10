//! Startup recovery: chunks the last run left open are finalized and handed back for the catalog.
//!
//! Killing the worker mid-chunk loses nothing already appended (every record goes straight to the file); it only
//! leaves the chunk without a footer. The next start scans each stream's newest chunks, finalizes those, and the
//! worker commits them like any closed chunk: each record carries the id of the payload it absorbed, so the chunk alone
//! says what to commit (a slot is marked `flushed` once its record was synced, so the broker may already have
//! reclaimed those slots; the chunk is the copy, and every record in it was durable or is being re-read from a slot
//! the broker still holds).
//!
//! Derived `*-copy-*.cnk` sidecars are recovered by `copies`, never absorbed as raw chunks.
//!
//! Leftovers of an interrupted [`crate::chunk::rewrite_without`] (`.rewrite` files) or Opus conversion
//! ([`crate::opus::pcm_chunk_to_opus`], `.partial` files) are deleted, since the original is still whole until the
//! rename; a chunk with no whole record at all is deleted too.
//!
//! Only unfinished chunks are finalized. A microphone chunk is converted to Opus after it is finished, by a rename
//! over its own file, so recovery never sees both a PCM and an Opus chunk for the same payloads: it sees the PCM
//! chunk still being written, or one finished file (PCM or Opus) that it leaves alone.

use std::fs;
use std::path::{Path, PathBuf};

use std::collections::{BTreeMap, BTreeSet};

use continuum_catalog::FrameProvenance;
use continuum_core::{StreamId, Uuid};

use crate::TierError;
use crate::chunk::{ChunkFormat, ChunkReader, Record, finalize, read_provenance};

/// A chunk found at startup without a footer, now finalized.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recovered {
    /// The chunk's file.
    pub path: PathBuf,
    /// Its stream.
    pub stream: StreamId,
    /// Its format.
    pub format: ChunkFormat,
    /// When it was created.
    pub created_ns: u64,
    /// The records recovered.
    pub records: Vec<Record>,
    /// Corrupt stretches the scan skipped.
    pub skipped: usize,
    /// What capture attested about its frames, by payload id, from the chunk's provenance records (empty for a
    /// chunk written without them).
    pub provenance: BTreeMap<Uuid, FrameProvenance>,
}

/// Finalizes every unfinished `.cnk` under `dir` and returns them, oldest first. Finished chunks are left alone, and
/// only each stream's chunks newer than its newest finished one are opened; files that aren't chunks are skipped;
/// empty chunks and `.rewrite` and `.partial` leftovers are deleted.
///
/// # Errors
///
/// An I/O error listing the directory or finalizing a chunk.
pub fn recover(dir: &Path) -> Result<Vec<Recovered>, TierError> {
    let mut out = Vec::new();
    if !dir.exists() {
        return Ok(out);
    }
    // Every stream's hour directories, newest first: a run finishes a stream's chunk before it opens the
    // next, so a stream's unfinished chunks are its newest, and past its first finished one nothing needs opening.
    // Opening every chunk in the archive to read its footer took minutes at each start, and the encoded cameras'
    // Tier 0 rings filled with unflushed frames meanwhile. Leftovers are still deleted from the listing.
    let paths: Vec<PathBuf> = crate::layout::chunk_paths(dir, None)?;
    let mut settled: BTreeSet<PathBuf> = BTreeSet::new(); // stream directories with a finished chunk seen
    for path in paths.into_iter().rev() {
        let ext = path.extension().and_then(|e| e.to_str());
        if matches!(ext, Some("rewrite" | "partial")) {
            fs::remove_file(&path)?;
            continue;
        }
        if ext != Some("cnk") || crate::copies::is_copy(&path) {
            continue;
        }
        let stream_dir = path.parent().and_then(Path::parent).map(Path::to_path_buf).unwrap_or_default();
        if settled.contains(&stream_dir) {
            continue;
        }
        let reader = match ChunkReader::open(&path) {
            Ok(r) => r,
            Err(crate::chunk::ChunkError::NotAChunk(_)) => continue,
            Err(e) => return Err(e.into()),
        };
        if reader.finalized() {
            settled.insert(stream_dir);
            continue;
        }
        if reader.records().is_empty() {
            drop(reader);
            fs::remove_file(&path)?;
            crate::layout::prune_hour_dir(&path);
            continue;
        }
        let (stream, format, created_ns, skipped) =
            (reader.stream(), reader.format(), reader.created_ns(), reader.skipped());
        drop(reader);
        let records = finalize(&path)?;
        let provenance = read_provenance(&path)?;
        out.push(Recovered { path, stream, format, created_ns, records, skipped, provenance });
    }
    out.reverse(); // oldest first, as the listing is
    Ok(out)
}
