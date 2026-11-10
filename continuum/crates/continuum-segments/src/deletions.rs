//! Forgetting rows, and the deletion feed.
//!
//! A forget is logical first and physical after: [`forget`] appends a deletion to `<root>/.deletions` (an append-only
//! log of JSON lines under a lock), and from then on every read, follow, and append leaves out the rows it names, so a
//! forgotten row is never shown again even before its file is rewritten and a late writer can't bring it back. It then
//! rewrites each closed part that holds such rows, as the part's next version; an open part is filtered as it's read
//! and rewritten when its writer closes it (or the next writer recovers it). Each deletion gets the next revision,
//! which [`revision`] and [`deletions`] give a reader that caches what it read or copied out of it: `forget`
//! invalidates copied content, `reset` only makes the evidence unavailable.

use std::fs::File;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use arrow_array::{Array, BooleanArray, FixedSizeBinaryArray, Int64Array, RecordBatch, StringArray};
use arrow_select::filter::filter_record_batch;
use serde::{Deserialize, Serialize};

use crate::{Error, Part, Result, TS, cameras, check_name, hour_of, io, parts};

/// What a deletion was for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// Someone asked for these rows to be forgotten: what was copied out of them must go too.
    Forget,
    /// The store was reset: the evidence is gone, but what was learned from it stands.
    Reset,
}

/// One deletion.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Deletion {
    /// Its place in the feed, from 1.
    pub revision: u64,
    /// What it was for.
    pub kind: Kind,
    /// The streams it covers; empty for every stream.
    pub streams: Vec<String>,
    /// The cameras it covers; empty for every camera.
    pub cameras: Vec<String>,
    /// Rows with `ts_ns` in `[t0, t1)`.
    pub t0: u64,
    /// The range's end.
    pub t1: u64,
    /// When it was made, UTC ns.
    pub at_ns: u64,
    /// With [`Deletion::value`], only rows whose column of this name holds that value (any time when the range is
    /// all of it): a person or voiceprint forgotten.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub column: Option<String>,
    /// The value: a UUID's string for an id column, the text for a str column.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
}

/// What one deletion leaves out of a stream's camera.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Filter {
    t0: u64,
    t1: u64,
    column: Option<(String, String)>,
}

impl Deletion {
    /// Whether it covers `stream`'s `camera`.
    #[must_use]
    pub fn covers(&self, stream: &str, camera: &str) -> bool {
        (self.streams.is_empty() || self.streams.iter().any(|s| s == stream))
            && (self.cameras.is_empty() || self.cameras.iter().any(|c| c == camera))
    }
}

fn log_path(root: &Path) -> PathBuf {
    root.join(".deletions")
}

/// The deletion log's length, 0 when there is none: it only grows, so a change means a new deletion.
pub(crate) fn log_len(root: &Path) -> u64 {
    std::fs::metadata(log_path(root)).map_or(0, |m| m.len())
}

/// Every deletion, oldest first.
///
/// # Errors
///
/// An I/O error, or [`Error::Invalid`] for a line that isn't a deletion.
pub fn all(root: &Path) -> Result<Vec<Deletion>> {
    let path = log_path(root);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(Error::Io(path, e)),
    };
    // A last line a crash cut short was never acknowledged: it isn't a deletion.
    let complete = text.rfind('\n').map_or("", |i| &text[..=i]);
    complete
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| Error::Invalid(format!("{}: {e}", path.display()))))
        .collect()
}

/// The newest deletion's revision; 0 when nothing was ever deleted.
///
/// # Errors
///
/// As [`all`].
pub fn revision(root: &Path) -> Result<u64> {
    Ok(all(root)?.last().map_or(0, |d| d.revision))
}

/// The deletions after `revision`, oldest first: what a reader that captured `revision` before reading must check.
///
/// # Errors
///
/// As [`all`].
pub fn deletions(root: &Path, revision: u64) -> Result<Vec<Deletion>> {
    Ok(all(root)?.into_iter().filter(|d| d.revision > revision).collect())
}

/// The deletions that cover a stream's camera, for filtering its rows.
pub(crate) fn covering(root: &Path, stream: &str, camera: &str) -> Result<Vec<Filter>> {
    Ok(filters(&all(root)?, stream, camera))
}

/// The filters of `deletions` (the log as read once) that cover a stream's camera.
pub(crate) fn filters(deletions: &[Deletion], stream: &str, camera: &str) -> Vec<Filter> {
    deletions
        .iter()
        .filter(|d| d.covers(stream, camera))
        .map(|d| Filter { t0: d.t0, t1: d.t1, column: d.column.clone().zip(d.value.clone()) })
        .collect()
}

/// The log as last read, for a reader that filters every poll: read again only when its length changed,
/// which every deletion appended does (the writer keys its own copy the same way), instead of parsing it for each
/// camera of each poll.
#[derive(Default)]
pub(crate) struct Cached {
    len: Option<u64>,
    all: Vec<Deletion>,
}

impl Cached {
    /// As [`covering`].
    pub(crate) fn covering(&mut self, root: &Path, stream: &str, camera: &str) -> Result<Vec<Filter>> {
        let len = log_len(root);
        if self.len != Some(len) {
            self.all = all(root)?;
            self.len = Some(len);
        }
        Ok(filters(&self.all, stream, camera))
    }
}

/// Whether row `i` of `column` holds `value` (a UUID string for a 16-byte id column, the text for a str column).
fn holds(column: &dyn Array, i: usize, value: &str, id: Option<[u8; 16]>) -> bool {
    if column.is_null(i) {
        return false;
    }
    if let Some(ids) = column.as_any().downcast_ref::<FixedSizeBinaryArray>() {
        return id.is_some_and(|id| ids.value(i) == id);
    }
    column.as_any().downcast_ref::<StringArray>().is_some_and(|s| s.value(i) == value)
}

fn uuid_bytes(s: &str) -> Option<[u8; 16]> {
    let hex: String = s.chars().filter(|c| *c != '-').collect();
    let mut out = [0u8; 16];
    if hex.len() != 32 {
        return None;
    }
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(hex.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(out)
}

/// A filter's column in a batch: the column, the value it names, and that value as an id's bytes.
type Named<'a> = (&'a dyn Array, &'a str, Option<[u8; 16]>);

/// `batch` without the rows any of `filters` names.
pub(crate) fn without(batch: RecordBatch, filters: &[Filter]) -> Result<RecordBatch> {
    if filters.is_empty() {
        return Ok(batch);
    }
    let ts = batch
        .column_by_name(TS)
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| Error::Invalid(format!("a batch without {TS}")))?;
    // Each filter's column, when it names one: a stream without that column has no rows it names.
    let columns: Vec<Option<Named<'_>>> = filters
        .iter()
        .map(|f| match &f.column {
            Some((name, value)) => batch.column_by_name(name).map(|c| (c.as_ref(), value.as_str(), uuid_bytes(value))),
            None => None,
        })
        .collect();
    #[allow(clippy::cast_sign_loss)]
    let keep: BooleanArray = (0..batch.num_rows())
        .map(|i| {
            let t = ts.value(i).max(0) as u64;
            let named = filters.iter().zip(&columns).any(|(f, c)| {
                t >= f.t0
                    && t < f.t1
                    && match (&f.column, c) {
                        (None, _) => true,
                        (Some(_), Some((col, value, id))) => holds(*col, i, value, *id),
                        (Some(_), None) => false,
                    }
            });
            Some(!named)
        })
        .collect();
    Ok(filter_record_batch(&batch, &keep)?)
}

/// Forgets `stream`s' (all when `None`) rows for `cams` (all when `None`) with `ts_ns` in `[t0, t1)`, and returns the
/// deletion. The rows are left out of every read from the moment this returns; closed parts holding them are
/// rewritten before it does.
///
/// # Errors
///
/// [`Error::Invalid`] for a bad name or an empty range, or an I/O or Arrow error.
pub fn forget(
    root: &Path,
    streams: Option<&[&str]>,
    cams: Option<&[&str]>,
    t0: u64,
    t1: u64,
    kind: Kind,
) -> Result<Deletion> {
    forget_matching(root, streams, cams, t0, t1, None, kind)
}

/// Forgets every row, at any time, of `streams` (all when `None`) and `cams` (all when `None`) whose `column` holds
/// `value`: a person's or a voiceprint's rows, with a UUID's string for an id column or the text for a str column.
/// As [`forget`], it takes effect at once and rewrites closed parts.
///
/// # Errors
///
/// As [`forget`], and [`Error::Invalid`] for an empty column name.
pub fn forget_where(
    root: &Path,
    streams: Option<&[&str]>,
    cams: Option<&[&str]>,
    column: &str,
    value: &str,
    kind: Kind,
) -> Result<Deletion> {
    if column.is_empty() || column == TS || column == crate::ROW_SEQ {
        return Err(Error::Invalid(format!("can't forget by column {column:?}")));
    }
    forget_matching(root, streams, cams, 0, u64::MAX, Some((column, value)), kind)
}

fn forget_matching(
    root: &Path,
    streams: Option<&[&str]>,
    cams: Option<&[&str]>,
    t0: u64,
    t1: u64,
    column: Option<(&str, &str)>,
    kind: Kind,
) -> Result<Deletion> {
    if t1 <= t0 {
        return Err(Error::Invalid("an empty range forgets nothing".to_owned()));
    }
    for s in streams.unwrap_or_default() {
        check_name("stream", s)?;
    }
    for c in cams.unwrap_or_default() {
        check_name("camera", c)?;
    }
    std::fs::create_dir_all(root).map_err(io(root))?;
    let deletion = append(root, |revision| Deletion {
        revision,
        kind,
        streams: streams.unwrap_or_default().iter().map(|&s| s.to_owned()).collect(),
        cameras: cams.unwrap_or_default().iter().map(|&c| c.to_owned()).collect(),
        t0,
        t1,
        at_ns: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)),
        column: column.map(|(c, _)| c.to_owned()),
        value: column.map(|(_, v)| v.to_owned()),
    })?;
    let all_streams: Vec<String> = match streams {
        Some(s) => s.iter().map(|&s| s.to_owned()).collect(),
        None => streams_of(root)?,
    };
    for stream in &all_streams {
        let cam_list: Vec<String> = match cams {
            Some(c) => c.iter().map(|&c| c.to_owned()).collect(),
            None => cameras(root, stream)?,
        };
        for cam in cam_list {
            let dir = root.join(stream).join(&cam);
            let ranges = covering(root, stream, &cam)?;
            crate::clear_stale(&dir)?;
            for p in parts(&dir)? {
                if p.end.is_some() && p.hour >= hour_of(t0) && p.hour <= hour_of(t1 - 1) {
                    purge_part(&p, &ranges)?;
                }
            }
        }
    }
    Ok(deletion)
}

/// Appends the next deletion under the log's lock, synced before it returns.
fn append(root: &Path, make: impl FnOnce(u64) -> Deletion) -> Result<Deletion> {
    let lock_path = root.join(".deletions.lock");
    let lock = File::options().create(true).truncate(false).write(true).open(&lock_path).map_err(io(&lock_path))?;
    lock.lock().map_err(io(&lock_path))?;
    let deletion = make(revision(root)? + 1);
    let path = log_path(root);
    // A last line a crash cut short is dropped first, so the new one starts a line of its own.
    if let Ok(text) = std::fs::read(&path)
        && text.last().is_some_and(|&b| b != b'\n')
    {
        let keep = text.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
        let f = File::options().write(true).open(&path).map_err(io(&path))?;
        f.set_len(keep as u64).map_err(io(&path))?;
    }
    let mut line = serde_json::to_string(&deletion).map_err(|e| Error::Invalid(e.to_string()))?;
    line.push('\n');
    let mut log = File::options().create(true).append(true).open(&path).map_err(io(&path))?;
    log.write_all(line.as_bytes()).map_err(io(&path))?;
    log.sync_data().map_err(io(&path))?;
    Ok(deletion)
}

/// The streams under `root`.
fn streams_of(root: &Path) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(root).map_err(io(root))? {
        let entry = entry.map_err(io(root))?;
        if let Some(name) = entry.file_name().to_str()
            && check_name("stream", name).is_ok()
            && entry.file_type().is_ok_and(|t| t.is_dir())
        {
            out.push(name.to_owned());
        }
    }
    out.sort();
    Ok(out)
}

/// Rewrites the part just closed at `path` without the rows the deletions covering its stream's camera name, when it
/// holds any: a forget while it was open only hid them.
pub(crate) fn purge_closed(root: &Path, stream: &str, camera: &str, path: &Path) -> Result<()> {
    let ranges = covering(root, stream, camera)?;
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let part = path.file_name().and_then(|f| f.to_str()).and_then(|f| Part::parse(dir, f));
    match part {
        Some(p) if !ranges.is_empty() => purge_part(&p, &ranges),
        _ => Ok(()),
    }
}

/// Rewrites a closed part without the rows `ranges` name, as its next version, when it holds any.
fn purge_part(p: &Part, ranges: &[Filter]) -> Result<()> {
    let batches = crate::reader::decode_file(&p.path)?;
    let before: usize = batches.iter().map(RecordBatch::num_rows).sum();
    let kept: Vec<RecordBatch> = batches.into_iter().map(|b| without(b, ranges)).collect::<Result<_>>()?;
    if kept.iter().map(RecordBatch::num_rows).sum::<usize>() == before {
        return Ok(());
    }
    let Some(end) = p.end else { return Ok(()) };
    let Some(schema) = kept.first().map(RecordBatch::schema) else { return Ok(()) };
    let dir = p.path.parent().unwrap_or_else(|| Path::new("."));
    let path = dir.join(Part::versioned_name(p.hour, p.part, p.first, end, p.version + 1));
    crate::replace_part(&path, &p.path, &schema, &kept)
}
