//! Where Tier 1 chunk files live under the chunk directory.
//!
//! Every chunk file, and every derived copy, is in `streams/<stream>/<YYYYMMDDHH>/` (the UTC hour it was created in;
//! a copy's by its payload's start), named `<stream>-<created_ns>-<n>.cnk` (`<stream>-copy-<payload-id>.cnk`). The
//! chunk directory holds nothing else: the earlier flat layout was dropped with its recordings.
//!
//! The flat directory held about 500,000 files on the Linux machine, about 1.85 new ones a second, and every listing
//! walked it (the gateway's stream listing, about every 20 s while Live is open: 1 to 11 s each). Walking by hour lets
//! a reader skip a directory that hasn't changed since it last read it: a finished hour's directory changes only when a
//! chunk in it is deleted.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use continuum_core::StreamId;

/// The subdirectory of the chunk directory that holds the per-stream, per-hour directories.
pub const STREAMS: &str = "streams";

/// The directory a chunk of `stream` created at `created_ns` (UTC nanoseconds) goes in:
/// `<dir>/streams/<stream>/<YYYYMMDDHH>`.
#[must_use]
pub fn hour_dir(dir: &Path, stream: StreamId, created_ns: u64) -> PathBuf {
    dir.join(STREAMS).join(stream.0.to_string()).join(hour_label(created_ns))
}

/// The UTC hour of `ns`, as `YYYYMMDDHH` (the segment store names its files the same way).
#[must_use]
pub fn hour_label(ns: u64) -> String {
    let secs = i64::try_from(ns / 1_000_000_000).unwrap_or(i64::MAX);
    chrono::DateTime::from_timestamp(secs, 0).map_or_else(|| "0000000000".into(), |t| t.format("%Y%m%d%H").to_string())
}

/// The catalog id of the chunk in the file named `<stream>-<created_ns>-<counter>.cnk`: a UUIDv7 at the file's
/// creation time whose other bits are a hash of the three, so it is the same every time it is derived. Every part of
/// a chunk names the same id, and a recovery after a crash asks the catalog what it holds under it
/// (docs/CONTINUUM-STORAGE.md).
#[must_use]
pub fn chunk_id(stream: StreamId, created_ns: u64, counter: u64) -> continuum_core::Uuid {
    // FNV-1a, written out: the id must not change with the compiler or a dependency.
    let hash = |seed: u64| {
        let mut h = 0xcbf2_9ce4_8422_2325u64 ^ seed;
        for b in
            u64::from(stream.0).to_le_bytes().into_iter().chain(created_ns.to_le_bytes()).chain(counter.to_le_bytes())
        {
            h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    };
    let mut rest = [0u8; 10];
    rest[..8].copy_from_slice(&hash(0).to_le_bytes());
    rest[8..].copy_from_slice(&hash(1).to_le_bytes()[..2]);
    continuum_core::Uuid(
        uuid::Builder::from_unix_timestamp_millis(created_ns / 1_000_000, &rest).into_uuid().into_bytes(),
    )
}

/// [`chunk_id`] for a chunk file, from its name; `None` for a name that isn't the flusher's.
#[must_use]
pub fn chunk_id_of(path: &Path) -> Option<continuum_core::Uuid> {
    if path.extension()? != "cnk" {
        return None;
    }
    let mut parts = path.file_stem()?.to_str()?.split('-');
    let (stream, created_ns, counter) =
        (parts.next()?.parse().ok()?, parts.next()?.parse().ok()?, parts.next()?.parse().ok()?);
    parts.next().is_none().then(|| chunk_id(StreamId(stream), created_ns, counter))
}

/// Every directory chunk files are in: each hour directory, of `stream` alone when one is given. Sorted; a chunk
/// directory that doesn't exist has none.
///
/// # Errors
///
/// An I/O error listing a directory, other than one that doesn't exist.
pub fn chunk_dirs(dir: &Path, stream: Option<StreamId>) -> io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let streams = dir.join(STREAMS);
    let stream_dirs = match stream {
        Some(s) => vec![streams.join(s.0.to_string())],
        None => subdirs(&streams)?.into_iter().filter(|p| is_stream_dir(p)).collect(),
    };
    for stream_dir in stream_dirs {
        out.extend(subdirs(&stream_dir)?);
    }
    out.sort();
    Ok(out)
}

/// The files directly in `dir` (none when it doesn't exist).
///
/// # Errors
///
/// An I/O error listing it, other than its not existing.
pub fn files_in(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut out = Vec::new();
    for entry in entries {
        let entry = entry?;
        if entry.file_type().is_ok_and(|t| t.is_file()) {
            out.push(entry.path());
        }
    }
    Ok(out)
}

/// Every file in the chunk directories (see [`chunk_dirs`]), of `stream`'s alone when one is given, sorted.
///
/// # Errors
///
/// An I/O error listing a directory, other than one that doesn't exist.
pub fn chunk_paths(dir: &Path, stream: Option<StreamId>) -> io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for d in chunk_dirs(dir, stream)? {
        out.extend(files_in(&d)?);
    }
    out.sort();
    Ok(out)
}

/// Makes `hour` (an [`hour_dir`] under `dir`) and the directories above it, each made durable in its parent: a chunk
/// in a directory whose own entry was lost in a crash would be lost with it. It syncs them even when they exist (a
/// crash may have come between making and syncing them): a writer calls it once per hour directory per run.
///
/// # Errors
///
/// An I/O error making or syncing a directory.
pub fn make_hour_dir(dir: &Path, hour: &Path) -> io::Result<()> {
    fs::create_dir_all(hour)?;
    let mut at = hour;
    while let Some(parent) = at.parent() {
        fs::File::open(parent)?.sync_all()?;
        if parent == dir {
            break;
        }
        at = parent;
    }
    Ok(())
}

/// After `file` was deleted: removes its hour directory when that's now empty, so the directories of hours aged out
/// whole don't pile up (about 170 a day with the 30-day video window) and every listing go on reading them. A
/// directory that isn't empty, or is gone already, is left; any other failure is ignored too, since an empty
/// directory left behind costs only a little listing time. A file outside the hour directories leaves its directory
/// alone.
pub fn prune_hour_dir(file: &Path) {
    let Some(hour) = file.parent() else { return };
    let in_streams = hour.parent().and_then(Path::parent).and_then(Path::file_name).is_some_and(|n| n == STREAMS);
    if in_streams {
        let _ = fs::remove_dir(hour); // fails, harmlessly, while the hour still holds a chunk
    }
}

fn subdirs(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut out = Vec::new();
    for entry in entries {
        let entry = entry?;
        if entry.file_type().is_ok_and(|t| t.is_dir()) {
            out.push(entry.path());
        }
    }
    Ok(out)
}

fn is_stream_dir(path: &Path) -> bool {
    path.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.parse::<u32>().is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOUR_NS: u64 = 3_600_000_000_000;
    // 2026-11-08T15:30:00Z
    const T: u64 = 1_794_151_800_000_000_000;

    #[test]
    fn a_chunk_goes_in_its_streams_utc_hour() {
        let dir = Path::new("/tier");
        assert_eq!(hour_dir(dir, StreamId(7), T), Path::new("/tier/streams/7/2026110815"));
        assert_eq!(hour_dir(dir, StreamId(7), T + HOUR_NS), Path::new("/tier/streams/7/2026110816"));
    }

    #[test]
    fn a_chunks_id_comes_from_its_name_alone() {
        let id = chunk_id(StreamId(7), T, 3);
        assert_eq!(chunk_id_of(Path::new(&format!("/tier/streams/7/2026110815/7-{T}-3.cnk"))), Some(id));
        assert_eq!(uuid::Uuid::from_bytes(id.0).get_version_num(), 7);
        let (secs, _) = uuid::Uuid::from_bytes(id.0).get_timestamp().expect("a v7 time").to_unix();
        assert_eq!(secs, T / 1_000_000_000, "it sorts by when the file was created");
        // Pinned: a recovery after an upgrade has to derive the id the last run committed under.
        assert_eq!(uuid::Uuid::from_bytes(id.0).to_string(), "01a1bbc7-dcc0-70c7-a624-e2f341a46101");
        for other in [chunk_id(StreamId(8), T, 3), chunk_id(StreamId(7), T + 1, 3), chunk_id(StreamId(7), T, 4)] {
            assert_ne!(other, id);
        }
        for name in ["7-copy-0190.cnk", "7-1-1.rejected", "7-1.cnk", "7-1-1-1.cnk", "cam-1-1.cnk"] {
            assert_eq!(chunk_id_of(Path::new(name)), None, "{name}");
        }
    }

    #[test]
    fn every_hour_is_listed_and_a_stream_alone_when_asked() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        fs::write(dir.join("7-1-1.cnk"), b"stray").unwrap(); // the chunk directory itself holds no chunks
        for (stream, at) in [(7, T), (7, T + HOUR_NS), (8, T)] {
            let hour = hour_dir(dir, StreamId(stream), at);
            make_hour_dir(dir, &hour).unwrap();
            fs::write(hour.join(format!("{stream}-{at}-1.cnk")), b"new").unwrap();
        }
        let names = |paths: Vec<PathBuf>| -> Vec<String> {
            paths.iter().map(|p| p.strip_prefix(dir).unwrap().to_string_lossy().into_owned()).collect()
        };
        assert_eq!(
            names(chunk_paths(dir, None).unwrap()),
            [
                format!("streams/7/2026110815/7-{T}-1.cnk"),
                format!("streams/7/2026110816/7-{}-1.cnk", T + HOUR_NS),
                format!("streams/8/2026110815/8-{T}-1.cnk"),
            ]
        );
        assert_eq!(names(chunk_paths(dir, Some(StreamId(8))).unwrap()), [format!("streams/8/2026110815/8-{T}-1.cnk")]);
    }

    #[test]
    fn an_hour_directory_goes_with_its_last_chunk_and_the_chunk_directory_stays() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let hour = hour_dir(dir, StreamId(7), T);
        make_hour_dir(dir, &hour).unwrap();
        let (a, b) = (hour.join("7-1-1.cnk"), hour.join("7-2-2.cnk"));
        fs::write(&a, b"a").unwrap();
        fs::write(&b, b"b").unwrap();
        fs::remove_file(&a).unwrap();
        prune_hour_dir(&a);
        assert!(hour.is_dir(), "it still holds a chunk");
        fs::remove_file(&b).unwrap();
        prune_hour_dir(&b);
        assert!(!hour.exists(), "its last chunk went");
        prune_hour_dir(&b); // gone already: nothing to do
        let stray = dir.join("7-0-0.cnk");
        fs::write(&stray, b"stray").unwrap();
        fs::remove_file(&stray).unwrap();
        prune_hour_dir(&stray);
        assert!(dir.is_dir(), "the chunk directory itself is never removed");
    }

    #[test]
    fn a_missing_chunk_directory_has_no_chunks() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(chunk_paths(&tmp.path().join("nowhere"), None).unwrap().is_empty());
    }
}
