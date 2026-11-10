//! Continuum's segment store: machine time series (detections, poses, motion, face rows, embeddings,
//! captions, coverage) as columnar files, one stream, one camera, and one UTC hour per file, outside the broker's
//! catalog.
//!
//! The filesystem is the index, so any number of processes can write and read with no shared database, and nothing is
//! rebuilt on start:
//!
//! ```text
//! <root>/<stream>/<camera>/<YYYYMMDDHH>-<part>-s<first>.arrows          open: its writer is appending
//! <root>/<stream>/<camera>/<YYYYMMDDHH>-<part>-s<first>-<end>.arrows    closed: rows first..end of row_seq
//! <root>/<stream>/<camera>/<YYYYMMDDHH>-<part>-s<first>-<end>-v<n>.arrows   the same part, rewritten n times
//! <root>/<stream>/<camera>/.writer.lock                                 held by the one writer
//! <root>/<stream>/<camera>/.seq                                         the next row_seq, kept past retention
//! ```
//!
//! A part is an Arrow IPC stream: the schema, then one record batch per [`Writer::append`], so a reader can read an
//! open part while it grows and stops cleanly at a torn tail. When a part closes it is rewritten as one batch (its
//! first rewrite, so a closed part normally ends `-v1`), and a read joins an open part's batches before it filters. Every row carries `ts_ns` (its time, UTC ns, which
//! decides its hour) and `row_seq` (per stream and camera, strictly increasing across hours and parts), which is what
//! a durable reader resumes from ([`Cursor`]). Rows for an hour whose part already closed (a late or imported row) go
//! to a new part of that hour, closed at once.
//!
//! Interval streams (tracks, visits, crossings, coverage) also carry `end_ns`, and [`read_overlapping`] finds every
//! row whose span reaches into a window, including one that began before it. Coverage is a stream like any other,
//! named `<stage>.coverage`, with `ts_ns`, `end_ns`, and a status column, written beside the stage's outputs.

mod deletions;
mod reader;
mod wake;
mod writer;

use std::path::{Path, PathBuf};

use arrow_array::RecordBatch;

pub use deletions::{Deletion, Kind as DeletionKind, deletions, forget, forget_where, revision};
pub use reader::{Follower, Followers, PartRef, Read, read, read_overlapping, tail};
pub use writer::Writer;

/// Every row's time, UTC nanoseconds: it decides the row's hour.
pub const TS: &str = "ts_ns";
/// A span's end, UTC nanoseconds, for streams whose rows are intervals (tracks, visits, coverage): what
/// [`read_overlapping`] reads by.
pub const END: &str = "end_ns";
/// Every row's sequence within its stream and camera, assigned by the writer.
pub const ROW_SEQ: &str = "row_seq";
/// Nanoseconds in an hour.
pub const HOUR_NS: u64 = 3_600_000_000_000;

/// What went wrong.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A file or directory operation.
    #[error("{0}: {1}")]
    Io(PathBuf, std::io::Error),
    /// An Arrow encode or decode.
    #[error(transparent)]
    Arrow(#[from] arrow_schema::ArrowError),
    /// Another process writes this stream and camera.
    #[error("another writer holds {0}")]
    Busy(PathBuf),
    /// A name, schema, or batch that doesn't fit the store.
    #[error("{0}")]
    Invalid(String),
}

/// Results of this crate.
pub type Result<T> = std::result::Result<T, Error>;

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> Error + '_ {
    move |e| Error::Io(path.to_path_buf(), e)
}

/// Where a durable reader is in one camera of a stream: the next `row_seq` it wants.
pub type Cursor = std::collections::BTreeMap<String, u64>;

/// A stream or camera name: letters, digits, `.`, `_`, and `-`, not starting with `.`, at most 128 bytes.
fn check_name(kind: &str, name: &str) -> Result<()> {
    let ok = !name.is_empty()
        && name.len() <= 128
        && !name.starts_with('.')
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'));
    if ok { Ok(()) } else { Err(Error::Invalid(format!("bad {kind} name {name:?}"))) }
}

/// The hour a time falls in, as hours since the epoch.
const fn hour_of(ts_ns: u64) -> u64 {
    ts_ns / HOUR_NS
}

/// An hour's `YYYYMMDDHH` (UTC).
fn hour_label(hour: u64) -> String {
    let secs = i64::try_from(hour * 3600).unwrap_or(i64::MAX);
    chrono::DateTime::from_timestamp(secs, 0)
        .map_or_else(|| "9999123123".to_owned(), |t| t.format("%Y%m%d%H").to_string())
}

fn hour_from_label(label: &str) -> Option<u64> {
    if label.len() != 10 || !label.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let t = chrono::NaiveDateTime::parse_from_str(&format!("{label}0000"), "%Y%m%d%H%M%S").ok()?;
    u64::try_from(t.and_utc().timestamp()).ok().map(|s| s / 3600)
}

/// One part's file, from its name.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Part {
    hour: u64,
    part: u32,
    first: u64,
    /// One past its last row's `row_seq`; `None` while it's open.
    end: Option<u64>,
    /// Bumped by every rewrite of a closed part (a forget, a compaction); 0 as first written.
    version: u32,
    path: PathBuf,
}

impl Part {
    fn name(hour: u64, part: u32, first: u64, end: Option<u64>) -> String {
        let label = hour_label(hour);
        match end {
            Some(end) => format!("{label}-{part:04}-s{first}-{end}.arrows"),
            None => format!("{label}-{part:04}-s{first}.arrows"),
        }
    }

    fn versioned_name(hour: u64, part: u32, first: u64, end: u64, version: u32) -> String {
        if version == 0 {
            return Self::name(hour, part, first, Some(end));
        }
        format!("{}-{part:04}-s{first}-{end}-v{version}.arrows", hour_label(hour))
    }

    fn parse(dir: &Path, file: &str) -> Option<Self> {
        let stem = file.strip_suffix(".arrows")?;
        let mut it = stem.split('-');
        let hour = hour_from_label(it.next()?)?;
        let part = it.next()?.parse().ok()?;
        let first = it.next()?.strip_prefix('s')?.parse().ok()?;
        let end = match it.next() {
            Some(e) => Some(e.parse().ok()?),
            None => None,
        };
        let version = match it.next() {
            Some(v) if end.is_some() => v.strip_prefix('v')?.parse().ok()?,
            Some(_) => return None,
            None => 0,
        };
        if it.next().is_some() || end.is_some_and(|e| e < first) {
            return None;
        }
        Some(Self { hour, part, first, end, version, path: dir.join(file) })
    }
}

/// A camera's parts, by hour then part number, each at its newest version. Files that aren't parts are ignored.
fn parts(dir: &Path) -> Result<Vec<Part>> {
    Ok(listing(dir)?.0)
}

/// A camera's parts as [`parts`] gives them, and the files of older versions a newer one replaced: what a crash
/// between a rewrite's rename and its removal of the old version leaves.
fn listing(dir: &Path) -> Result<(Vec<Part>, Vec<PathBuf>)> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), Vec::new())),
        Err(e) => return Err(Error::Io(dir.to_path_buf(), e)),
    };
    let mut all = Vec::new();
    for entry in entries {
        let entry = entry.map_err(io(dir))?;
        if let Some(p) = entry.file_name().to_str().and_then(|f| Part::parse(dir, f)) {
            all.push(p);
        }
    }
    // Closed after open, then by version, so the last of each part (its hour, number, and first row) is the one to read.
    all.sort_by_key(|p| (p.hour, p.part, p.first, p.end.is_some(), p.version));
    let mut out: Vec<Part> = Vec::with_capacity(all.len());
    let mut stale = Vec::new();
    for p in all {
        match out.last_mut() {
            Some(last) if (last.hour, last.part, last.first) == (p.hour, p.part, p.first) && last.end.is_some() => {
                stale.push(std::mem::replace(last, p).path);
            }
            _ => out.push(p),
        }
    }
    Ok((out, stale))
}

/// Removes the older versions [`listing`] found. One already gone (another process cleared it) is fine.
fn clear_stale(dir: &Path) -> Result<()> {
    for path in listing(dir)?.1 {
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(Error::Io(path, e)),
        }
    }
    Ok(())
}

/// Writes a closed part's next version to `path` whole or not at all, then removes `old`: the batches go to a
/// temporary name no listing reads, which is synced and renamed into place, and the directory synced, before the old
/// version goes. A crash leaves the old version alone, or both (a listing takes the newer), never a torn part.
fn replace_part(path: &Path, old: &Path, schema: &arrow_schema::SchemaRef, batches: &[RecordBatch]) -> Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(format!(".{}-{}.tmp", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    let tmp = PathBuf::from(tmp);
    let written = (|| {
        let file = std::fs::File::options().create_new(true).write(true).open(&tmp).map_err(io(&tmp))?;
        let options = arrow_ipc::writer::IpcWriteOptions::default()
            .try_with_compression(Some(arrow_ipc::CompressionType::ZSTD))?;
        let mut writer = arrow_ipc::writer::StreamWriter::try_new_with_options(file, schema, options)?;
        for batch in batches.iter().filter(|b| b.num_rows() > 0) {
            writer.write(batch)?;
        }
        writer.finish()?;
        writer.get_ref().sync_data().map_err(io(&tmp))?;
        std::fs::rename(&tmp, path).map_err(io(path))
    })();
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::File::open(dir).and_then(|d| d.sync_all()).map_err(io(dir))?;
    match std::fs::remove_file(old) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(Error::Io(old.to_path_buf(), e)),
    }
}

/// Every row of `batches` as one batch, in their order; `None` when there are none.
fn joined(batches: &[RecordBatch]) -> Result<Option<RecordBatch>> {
    match batches {
        [] => Ok(None),
        [one] => Ok(Some(one.clone())),
        [first, ..] => Ok(Some(arrow_select::concat::concat_batches(&first.schema(), batches)?)),
    }
}

/// Rewrites the closed part at `path` as one record batch, as its next version, and returns where it is now. A part
/// is written a batch an append, so a reader can follow it while it's open; a stream appended a row at a time (a
/// coverage span a second) left an hour as thousands of one-row batches, each with its own framing, and reading a
/// day of it meant decoding hundreds of thousands of them. A part already in one batch, or empty, stays as it is.
fn compact_closed(path: &Path) -> Result<PathBuf> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let Some(p) = path.file_name().and_then(|f| f.to_str()).and_then(|f| Part::parse(dir, f)) else {
        return Ok(path.to_path_buf());
    };
    let Some(end) = p.end else { return Ok(path.to_path_buf()) };
    let batches = reader::decode_file(path)?;
    if batches.len() < 2 {
        return Ok(path.to_path_buf());
    }
    let Some(one) = joined(&batches)? else { return Ok(path.to_path_buf()) };
    let next = dir.join(Part::versioned_name(p.hour, p.part, p.first, end, p.version + 1));
    replace_part(&next, path, &one.schema(), &[one])?;
    Ok(next)
}

/// Rewrites every closed part under `root` that holds more than one batch as one ([`Writer`] does this as it closes
/// a part; this is for parts closed before it did, or left by a crash between a close and its rewrite), without the
/// rows a forget names. Returns how many parts it rewrote. Safe beside running writers and readers: it touches closed
/// parts only, the way a forget does.
///
/// # Errors
///
/// An I/O or Arrow error.
pub fn compact(root: &Path) -> Result<u64> {
    let mut rewritten = 0;
    for stream in deletions::streams_of(root)? {
        for cam in cameras(root, &stream)? {
            let dir = root.join(&stream).join(&cam);
            let ranges = deletions::covering(root, &stream, &cam)?;
            for p in parts(&dir)? {
                let Some(end) = p.end else { continue };
                let batches = match reader::decode_file(&p.path) {
                    Ok(batches) => batches,
                    // Rewritten or removed since the listing: whoever did has it.
                    Err(Error::Io(_, e)) if e.kind() == std::io::ErrorKind::NotFound => continue,
                    Err(e) => return Err(e),
                };
                if batches.len() < 2 {
                    continue;
                }
                let Some(one) = joined(&batches)? else { continue };
                let one = deletions::without(one, &ranges)?;
                let next = dir.join(Part::versioned_name(p.hour, p.part, p.first, end, p.version + 1));
                replace_part(&next, &p.path, &one.schema(), &[one])?;
                rewritten += 1;
            }
        }
    }
    Ok(rewritten)
}

/// The cameras a stream has parts for.
fn cameras(root: &Path, stream: &str) -> Result<Vec<String>> {
    let dir = root.join(stream);
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(Error::Io(dir, e)),
    };
    let mut out = Vec::new();
    for entry in entries {
        let entry = entry.map_err(io(&dir))?;
        if let Some(name) = entry.file_name().to_str()
            && check_name("camera", name).is_ok()
            && entry.file_type().is_ok_and(|t| t.is_dir())
        {
            out.push(name.to_owned());
        }
    }
    out.sort();
    Ok(out)
}

#[cfg(test)]
mod tests;
