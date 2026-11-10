//! Opaque derived payload copies: one ordinary passthrough CNK record, never an absorbed raw chunk.
//!
//! The original header/provenance stays in the catalog. A deterministic filename allows startup to attach an
//! already durable file to that same ID after a crash between fsync and the broker acknowledgement. Temporary
//! files have never been attached; recovery removes them. Copy files are excluded from raw chunk recovery, and purges
//! delete the whole copy instead of sending RewriteChunk for the original payload.
//!
//! Once the broker acknowledges a copy, [`mark_committed`] sets the file's modification time to the Unix epoch, and
//! recovery skips marked files after a stat. Before, every start reopened, synced, and committed every copy again,
//! one catalog write each, so starts slowed as copies piled up. The mark is set
//! without a sync, so a crash can lose it, and then that copy is just committed again; it is only set after the
//! acknowledgement, so a marked copy is never one the broker lacks. It doesn't depend on the clock, as a "committed
//! up to this time" watermark would: a clock stepped back could make that skip an uncommitted copy.

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use continuum_core::{HeaderBuf, StorageLocator, StorageTier, StreamId, Uuid};

use crate::TierError;
use crate::chunk::{ChunkFormat, ChunkReader, ChunkWriter, Record};

#[derive(Debug)]
pub(crate) struct Copy {
    pub path: PathBuf,
    pub stream: StreamId,
    pub record: Record,
    pub location: StorageLocator,
}

fn identity(path: &Path) -> Option<(StreamId, Uuid)> {
    let (stream, id) = path.file_stem()?.to_str()?.split_once("-copy-")?;
    Some((StreamId(stream.parse().ok()?), Uuid(uuid::Uuid::parse_str(id).ok()?.into_bytes())))
}

pub(crate) fn is_copy(path: &Path) -> bool {
    path.extension().is_some_and(|ext| ext == "cnk")
        && path
            .file_stem()
            .and_then(|name| name.to_str())
            .and_then(|name| name.split_once("-copy-"))
            .is_some_and(|(stream, _)| stream.parse::<u32>().is_ok())
}

fn sync_dir(dir: &Path) -> Result<(), TierError> {
    File::open(dir)?.sync_all()?;
    Ok(())
}

pub(crate) fn remove(path: &Path) -> Result<(), TierError> {
    match fs::remove_file(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    sync_dir(path.parent().unwrap_or(Path::new(".")))
}

fn open(path: &Path) -> Result<Copy, TierError> {
    let Some((stream, id)) = identity(path) else {
        return Err(TierError::Codec(format!("{}: invalid payload-copy identity", path.display())));
    };
    let reader = ChunkReader::open(path)?;
    let record = reader.records().first().cloned();
    if reader.stream() != stream
        || reader.format() != ChunkFormat::Passthrough
        || !reader.finalized()
        || reader.skipped() != 0
        || reader.records().len() != 1
        || record.as_ref().is_none_or(|record| record.id != id || record.cut)
    {
        return Err(TierError::Codec(format!("{}: invalid single-record payload copy", path.display())));
    }
    let location = StorageLocator::Chunk {
        tier: StorageTier::Tier1Warm,
        uri: path.to_string_lossy().into_owned(),
        offset: 0,
        length: fs::metadata(path)?.len(),
    };
    Ok(Copy { path: path.to_path_buf(), stream, record: record.expect("checked"), location })
}

pub(crate) fn write(dir: &Path, header: &HeaderBuf, bytes: &[u8]) -> Result<Copy, TierError> {
    // In the hour directory of the payload's own start, so a retry finds the same file.
    let hour = crate::layout::hour_dir(dir, header.stream(), header.start_ns());
    crate::layout::make_hour_dir(dir, &hour)?;
    let path = hour.join(format!("{}-copy-{}.cnk", header.stream().0, header.id()));
    if path.exists() {
        let found = open(&path)?;
        let record = &found.record;
        let mut reader = ChunkReader::open(&path)?;
        if (record.start_ns, record.end_ns, record.seq) != (header.start_ns(), header.end_ns(), header.seq())
            || reader.read(0)? != bytes
        {
            return Err(TierError::Codec(format!("{}: conflicting payload-copy retry", path.display())));
        }
        // A retry may follow a crash immediately after rename: prove durability again before attachment.
        File::open(&path)?.sync_all()?;
        sync_dir(&hour)?;
        return Ok(found);
    }
    let temporary = path.with_extension("copy-write");
    if temporary.exists() {
        remove(&temporary)?;
    }
    let mut writer = ChunkWriter::create(&temporary, header.stream(), ChunkFormat::Passthrough, header.start_ns())?;
    writer.append_payload(header.id(), header.start_ns(), header.end_ns(), header.seq(), bytes)?;
    writer.close()?; // fsync the complete CRC-protected container before publishing its name
    fs::rename(&temporary, &path)?;
    sync_dir(&hour)?;
    open(&path)
}

/// The modification time of a copy the broker has acknowledged (see the module doc).
const COMMITTED: SystemTime = UNIX_EPOCH;

/// Marks `path` as a copy the broker acknowledged, so recovery passes over it. Not synced: a lost mark only means the
/// copy is committed again at the next start.
pub(crate) fn mark_committed(path: &Path) -> Result<(), TierError> {
    File::open(path)?.set_modified(COMMITTED)?;
    Ok(())
}

fn committed(path: &Path) -> Result<bool, TierError> {
    Ok(fs::metadata(path)?.modified()? == COMMITTED)
}

/// The published copies under `dir` that may not have been acknowledged (unmarked), each synced again for the caller
/// to commit; interrupted writes are removed.
pub(crate) fn recover(dir: &Path) -> Result<Vec<Copy>, TierError> {
    let mut found = Vec::new();
    let paths = crate::layout::chunk_paths(dir, None)?;
    for path in paths {
        if path.extension().is_some_and(|ext| ext == "copy-write") && identity(&path).is_some() {
            remove(&path)?;
        } else if is_copy(&path) && !committed(&path)? {
            let copy = open(&path)?;
            File::open(&path)?.sync_all()?;
            sync_dir(path.parent().unwrap_or(dir))?;
            found.push(copy);
        }
    }
    Ok(found)
}

pub(crate) fn purge(path: &Path, ids: &std::collections::BTreeSet<Uuid>) -> Result<bool, TierError> {
    let copy = match open(path) {
        Ok(copy) => copy,
        Err(TierError::Chunk(crate::ChunkError::Io(error)) | TierError::Io(error))
            if error.kind() == std::io::ErrorKind::NotFound =>
        {
            sync_dir(path.parent().unwrap_or(Path::new(".")))?;
            return Ok(false);
        }
        Err(error) => return Err(error),
    };
    // Notice spans describe what was removed, not every overlapping derived original. The client gathers all
    // notice pages before delivery, so only the complete, authoritative ID set decides which sidecars to delete.
    if ids.contains(&copy.record.id) {
        remove(path)?;
        return Ok(true);
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_core::HeaderSpec;

    fn header(stream: StreamId, parent: u8, subject: u8) -> HeaderBuf {
        let mut spec = HeaderSpec::raw(Uuid(uuid::Uuid::now_v7().into_bytes()), stream, 100, 200);
        spec.layer = continuum_core::DomainLayer::Layer3Interpretation;
        spec.parents = vec![Uuid([parent; 16])];
        spec.subjects = vec![Uuid([subject; 16])];
        HeaderBuf::build(&spec).unwrap()
    }

    #[test]
    fn copy_purge_preserves_unrelated_overlapping_payload_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let removed = header(StreamId(7), 1, 11);
        let sibling = header(StreamId(7), 2, 22);
        let first = write(dir.path(), &removed, b"removed").unwrap();
        let second = write(dir.path(), &sibling, b"unrelated").unwrap();
        let ids = [removed.id()].into_iter().collect();
        assert!(purge(&first.path, &ids).unwrap());
        assert!(!purge(&second.path, &ids).unwrap(), "notice spans are hints, not authority over sibling payloads");
        assert_eq!(ChunkReader::open(&second.path).unwrap().read(0).unwrap(), b"unrelated");
    }
    #[test]
    fn copy_retry_reuses_the_original_file_and_rejects_different_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let original = header(StreamId(7), 1, 11);
        let first = write(dir.path(), &original, b"original").unwrap();
        let bytes = fs::read(&first.path).unwrap();
        let second = write(dir.path(), &original, b"original").unwrap();
        assert_eq!(first.location, second.location);
        assert!(write(dir.path(), &original, b"replacement").is_err());
        assert_eq!(fs::read(&first.path).unwrap(), bytes);
        assert_eq!(recover(dir.path()).unwrap().len(), 1);
    }

    #[test]
    fn crash_recovery_reattaches_published_copies_but_never_raw_absorbs_them() {
        let dir = tempfile::tempdir().unwrap();
        let original = header(StreamId(7), 1, 11);
        let copy = write(dir.path(), &original, b"original").unwrap();
        let temporary = copy.path.with_extension("copy-write");
        fs::write(&temporary, b"interrupted write").unwrap();
        assert!(crate::recover(dir.path()).unwrap().is_empty());
        let recovered = recover(dir.path()).unwrap();
        assert_eq!(recovered.len(), 1);
        assert_eq!((recovered[0].record.id, recovered[0].stream), (original.id(), original.stream()));
        assert_eq!(recovered[0].location, copy.location);
        assert!(!temporary.exists());
        assert!(purge(&copy.path, &[original.id()].into_iter().collect()).unwrap());
        assert!(!purge(&copy.path, &[original.id()].into_iter().collect()).unwrap());
    }

    #[test]
    fn recovery_returns_only_the_copies_not_yet_marked_committed() {
        let dir = tempfile::tempdir().unwrap();
        let committed_header = header(StreamId(7), 1, 11);
        let pending_header = header(StreamId(7), 2, 22);
        let done = write(dir.path(), &committed_header, b"acknowledged").unwrap();
        mark_committed(&done.path).unwrap();
        write(dir.path(), &pending_header, b"in flight").unwrap();
        let recovered = recover(dir.path()).unwrap();
        assert_eq!(recovered.iter().map(|copy| copy.record.id).collect::<Vec<_>>(), vec![pending_header.id()]);
        assert_eq!(
            ChunkReader::open(&done.path).unwrap().read(0).unwrap(),
            b"acknowledged",
            "the mark keeps the bytes"
        );
    }

    #[test]
    fn a_copy_whose_mark_was_lost_is_recovered_again() {
        let dir = tempfile::tempdir().unwrap();
        let original = header(StreamId(7), 1, 11);
        let copy = write(dir.path(), &original, b"original").unwrap();
        mark_committed(&copy.path).unwrap();
        assert!(recover(dir.path()).unwrap().is_empty());
        File::open(&copy.path).unwrap().set_modified(SystemTime::now()).unwrap(); // a crash before the mark was durable
        let recovered = recover(dir.path()).unwrap();
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].location, copy.location);
    }

    #[test]
    fn mismatched_copy_identity_is_not_guessed_or_absorbed_during_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let original = header(StreamId(7), 1, 11);
        let copy = write(dir.path(), &original, b"original").unwrap();
        let wrong = copy.path.with_file_name(format!("8-copy-{}.cnk", original.id()));
        fs::rename(&copy.path, &wrong).unwrap();
        assert!(crate::recover(dir.path()).unwrap().is_empty());
        assert!(recover(dir.path()).is_err());
        assert!(wrong.exists(), "invalid copies remain available for diagnosis, never silently discarded");
    }
}
