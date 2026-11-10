//! Reading parts: a time range across cameras ([`read`]), and following new rows from a cursor ([`Follower`]),
//! open parts included as they grow.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read as _, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use arrow_array::{Array, BooleanArray, Int64Array, RecordBatch, UInt64Array};
use arrow_buffer::Buffer;
use arrow_ipc::reader::StreamDecoder;
use arrow_select::filter::filter_record_batch;

use crate::wake::Wake;
use crate::{Cursor, Error, Part, ROW_SEQ, Result, TS, cameras, check_name, hour_of, io, parts};

/// Every complete batch in a part's file, stopping at a torn tail (an open part's batch still being written, or one
/// a dead writer left half done).
pub(crate) fn decode_file(path: &Path) -> Result<Vec<RecordBatch>> {
    let bytes = std::fs::read(path).map_err(io(path))?;
    let mut decoder = StreamDecoder::new();
    Ok(decode(&mut decoder, bytes))
}

fn decode(decoder: &mut StreamDecoder, bytes: Vec<u8>) -> Vec<RecordBatch> {
    let mut buffer = Buffer::from_vec(bytes);
    let mut out = Vec::new();
    while !buffer.is_empty() {
        match decoder.decode(&mut buffer) {
            Ok(Some(batch)) => out.push(batch),
            Ok(None) | Err(_) => break,
        }
    }
    out
}

fn column<'a, T: 'static>(batch: &'a RecordBatch, name: &str) -> Result<&'a T> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<T>())
        .ok_or_else(|| Error::Invalid(format!("a part without {name}")))
}

/// The cameras asked for, or every camera the stream has.
fn wanted(root: &Path, stream: &str, cams: Option<&[&str]>) -> Result<Vec<String>> {
    check_name("stream", stream)?;
    match cams {
        Some(c) => {
            for cam in c {
                check_name("camera", cam)?;
            }
            Ok(c.iter().map(|&s| s.to_owned()).collect())
        }
        None => cameras(root, stream),
    }
}

/// One part a read saw: together they identify exactly what it read, for a cache to key on.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PartRef {
    /// Its camera.
    pub camera: String,
    /// Its hour, as hours since the epoch.
    pub hour: u64,
    /// Its number within the hour.
    pub part: u32,
    /// Its first row's `row_seq`.
    pub first: u64,
    /// One past its last row's `row_seq` once it's closed; `None` while it's open.
    pub end: Option<u64>,
    /// How many times it was rewritten.
    pub version: u32,
    /// The rows read from it (all of a closed part's; an open one's so far).
    pub rows: u64,
}

/// What [`read`] found: the rows, and the parts they came from.
#[derive(Debug, Default)]
pub struct Read {
    /// (camera, batch), in camera, hour, and part order.
    pub batches: Vec<(String, RecordBatch)>,
    /// Every part in the range, rows or not: a new part, a rewrite, or an open part's growth changes it.
    pub parts: Vec<PartRef>,
    /// The deletion feed's revision captured before reading: a deletion after it may name rows this read returned.
    pub revision: u64,
}

/// Every row of `stream` with `ts_ns` in `[t0, t1)`, for `cams` (every camera when `None`). Open parts are read as
/// far as they're written.
///
/// # Errors
///
/// [`Error::Invalid`] for a bad name, or an I/O or Arrow error.
pub fn read(root: &Path, stream: &str, cams: Option<&[&str]>, t0: u64, t1: u64) -> Result<Read> {
    read_window(root, stream, cams, t0, t1, false)
}

/// Every row of `stream` whose span reaches into `[t0, t1)`: `ts_ns < t1` and `end_ns > t0` (an instant row, by its
/// `ts_ns`), so a track or visit that began before `t0` is found too. Rows without `end_ns` are read as instants.
///
/// # Errors
///
/// As [`read`].
pub fn read_overlapping(root: &Path, stream: &str, cams: Option<&[&str]>, t0: u64, t1: u64) -> Result<Read> {
    read_window(root, stream, cams, t0, t1, true)
}

fn read_window(root: &Path, stream: &str, cams: Option<&[&str]>, t0: u64, t1: u64, overlap: bool) -> Result<Read> {
    // The log read once for the revision and every camera's filters (#3210), so both come from the same snapshot.
    let log = crate::deletions::all(root)?;
    let mut out = Read { revision: log.last().map_or(0, |d| d.revision), ..Read::default() };
    if t1 <= t0 {
        return Ok(out);
    }
    for cam in wanted(root, stream, cams)? {
        let (batches, parts) = (out.batches.len(), out.parts.len());
        let mut tries = 0;
        loop {
            let forgotten = crate::deletions::filters(&log, stream, &cam);
            match read_camera(root, stream, &cam, (t0, t1), overlap, &forgotten, &mut out) {
                // A part rewritten (or closed) between the listing and its read: list again.
                Err(Error::Io(_, e)) if e.kind() == std::io::ErrorKind::NotFound && tries < RELIST_TRIES => {
                    out.batches.truncate(batches);
                    out.parts.truncate(parts);
                    tries += 1;
                }
                done => break done?,
            }
        }
    }
    Ok(out)
}

/// How many times a read lists a camera's parts again when one went between the listing and its read.
const RELIST_TRIES: usize = 3;

/// One camera's part of [`read_window`], added to `out`.
fn read_camera(
    root: &Path,
    stream: &str,
    cam: &str,
    (t0, t1): (u64, u64),
    overlap: bool,
    forgotten: &[crate::deletions::Filter],
    out: &mut Read,
) -> Result<()> {
    let dir = root.join(stream).join(cam);
    // A span that began up to the camera's longest span before t0 can reach into the window.
    let reach = if overlap { crate::writer::read_u64(&dir.join(".maxspan"))?.unwrap_or(0) } else { 0 };
    let (first_hour, last_hour) = (hour_of(t0.saturating_sub(reach)), hour_of(t1 - 1));
    for p in parts(&dir)? {
        if p.hour < first_hour || p.hour > last_hour {
            continue;
        }
        let mut rows = 0;
        for batch in decode_file(&p.path)? {
            rows += batch.num_rows() as u64;
            let ts = column::<Int64Array>(&batch, TS)?;
            let end = if overlap {
                batch.column_by_name(crate::END).and_then(|c| c.as_any().downcast_ref::<Int64Array>())
            } else {
                None
            };
            #[allow(clippy::cast_sign_loss)]
            let keep: BooleanArray = (0..batch.num_rows())
                .map(|i| {
                    let t = ts.value(i).max(0) as u64;
                    let e = end.map_or(t, |e| e.value(i).max(0) as u64);
                    Some(if overlap { t < t1 && (e > t0 || (e <= t && t >= t0)) } else { t >= t0 && t < t1 })
                })
                .collect();
            let kept = crate::deletions::without(filter_record_batch(&batch, &keep)?, forgotten)?;
            if kept.num_rows() > 0 {
                out.batches.push((cam.to_owned(), kept));
            }
        }
        out.parts.push(PartRef {
            camera: cam.to_owned(),
            hour: p.hour,
            part: p.part,
            first: p.first,
            end: p.end,
            version: p.version,
            rows,
        });
    }
    Ok(())
}

struct Tail {
    decoder: StreamDecoder,
    offset: u64,
    done: bool,
}

/// Follows a stream's rows from a [`Cursor`]: each [`Follower::poll`] returns the rows written since, per camera in
/// `row_seq` order, and moves the cursor past them. Parts are found by listing, so a new hour, a late part, and a
/// part closed (renamed) under it are all picked up; an open part is read from where the last poll stopped.
pub struct Follower {
    root: std::path::PathBuf,
    stream: String,
    cams: Option<Vec<String>>,
    cursor: Cursor,
    tails: BTreeMap<(String, u64, u32), Tail>,
    deletions: crate::deletions::Cached,
    wake: Option<Wake>,
}

impl Follower {
    /// Follows `stream` for `cams` (every camera, those that appear later included, when `None`) from `cursor`: a
    /// camera missing from it starts at its first row.
    #[must_use]
    pub fn new(root: &Path, stream: &str, cams: Option<Vec<String>>, cursor: Cursor) -> Self {
        Self {
            root: root.to_path_buf(),
            stream: stream.to_owned(),
            cams,
            cursor,
            tails: BTreeMap::new(),
            deletions: crate::deletions::Cached::default(),
            wake: None,
        }
    }

    /// The directories its rows come in through: the stream's, and each followed camera's (for every camera, the
    /// ones there now; the stream's directory wakes a wait when another is made).
    fn dirs(&self) -> Vec<PathBuf> {
        let stream = self.root.join(&self.stream);
        let cams = match &self.cams {
            Some(c) => c.clone(),
            None => cameras(&self.root, &self.stream).unwrap_or_default(),
        };
        let mut out: Vec<PathBuf> = cams.iter().map(|c| stream.join(c)).collect();
        out.push(stream);
        out
    }

    /// Where it is: save this to resume.
    #[must_use]
    pub const fn cursor(&self) -> &Cursor {
        &self.cursor
    }

    /// The rows written since the last poll, as (camera, batch), each camera's in `row_seq` order.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a bad name, or an I/O or Arrow error.
    pub fn poll(&mut self) -> Result<Vec<(String, RecordBatch)>> {
        let cams = match &self.cams {
            Some(c) => c.clone(),
            None => wanted(&self.root, &self.stream, None)?,
        };
        let mut out = Vec::new();
        for cam in cams {
            let next = self.cursor.get(&cam).copied().unwrap_or(0);
            let forgotten = self.deletions.covering(&self.root, &self.stream, &cam)?;
            let mut got: Vec<RecordBatch> = Vec::new();
            for p in parts(&self.root.join(&self.stream).join(&cam))? {
                if p.end.is_some_and(|end| end <= next) {
                    continue; // all behind the cursor
                }
                got.extend(self.tail(&cam, &p)?);
            }
            let mut batches = Vec::new();
            for batch in got {
                let seq = column::<UInt64Array>(&batch, ROW_SEQ)?;
                let keep: BooleanArray = seq.values().iter().map(|&s| Some(s >= next)).collect();
                let kept = filter_record_batch(&batch, &keep)?;
                if kept.num_rows() == 0 {
                    continue;
                }
                // The cursor passes forgotten rows too, so a later poll doesn't look for them again.
                let seqs = column::<UInt64Array>(&kept, ROW_SEQ)?;
                let last = seqs.value(seqs.len() - 1);
                let first = seqs.value(0);
                let kept = crate::deletions::without(kept, &forgotten)?;
                if kept.num_rows() > 0 {
                    batches.push((first, kept));
                } else {
                    let at = self.cursor.entry(cam.clone()).or_insert(0);
                    *at = (*at).max(last + 1);
                }
            }
            batches.sort_by_key(|(first, _)| *first);
            for (_, batch) in batches {
                let seq = column::<UInt64Array>(&batch, ROW_SEQ)?;
                let last = seq.value(seq.len() - 1);
                let at = self.cursor.entry(cam.clone()).or_insert(0);
                *at = (*at).max(last + 1);
                out.push((cam.clone(), batch));
            }
        }
        // A part retention removed is forgotten with it.
        self.tails.retain(|_, t| !t.done);
        Ok(out)
    }

    /// [`Follower::poll`], waiting up to `timeout` for rows when there are none yet, off any lock of the caller's.
    /// On Linux it sleeps until a part in its directories is written, made, or renamed (#3211); elsewhere it checks
    /// again every [`WAIT_STEP`].
    ///
    /// # Errors
    ///
    /// As [`Follower::poll`].
    pub fn wait(&mut self, timeout: std::time::Duration) -> Result<Vec<(String, RecordBatch)>> {
        if timeout.is_zero() {
            return self.poll(); // nothing to wait for: no descriptor, no watches
        }
        let deadline = deadline_after(timeout);
        let mut wake = self.wake.take().unwrap_or_else(Wake::new);
        let got = loop {
            wake.watch(&self.root, self.dirs().iter().map(PathBuf::as_path));
            let got = match self.poll() {
                Ok(got) => got,
                Err(e) => break Err(e),
            };
            let now = std::time::Instant::now();
            if !got.is_empty() || now >= deadline {
                break Ok(got);
            }
            wake.sleep(deadline - now);
        };
        self.wake = Some(wake);
        got
    }

    /// The complete batches of a part not yet read.
    fn tail(&mut self, cam: &str, p: &Part) -> Result<Vec<RecordBatch>> {
        let key = (cam.to_owned(), p.hour, p.part);
        let tail =
            self.tails.entry(key).or_insert_with(|| Tail { decoder: StreamDecoder::new(), offset: 0, done: false });
        if tail.done {
            return Ok(Vec::new());
        }
        let mut file = match File::open(&p.path) {
            Ok(f) => f,
            // Renamed closed between the listing and here: the next poll finds it under its new name.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(Error::Io(p.path.clone(), e)),
        };
        file.seek(SeekFrom::Start(tail.offset)).map_err(io(&p.path))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).map_err(io(&p.path))?;
        tail.offset += bytes.len() as u64;
        let batches = decode(&mut tail.decoder, bytes);
        // A closed part read to its end has nothing more.
        tail.done = p.end.is_some();
        Ok(batches)
    }
}

/// When a wait of `timeout` from now ends: about a century from now for one too long to add (Python's
/// `poll(timeout_s)` takes up to about 1.8e19 s, and `Instant + Duration` panics past about 9.2e18 s).
fn deadline_after(timeout: std::time::Duration) -> std::time::Instant {
    let now = std::time::Instant::now();
    now.checked_add(timeout).unwrap_or_else(|| now + std::time::Duration::from_secs(100 * 365 * 86_400))
}

/// How often [`Follower::wait`] looks again for rows when it can't watch its directories.
pub const WAIT_STEP: std::time::Duration = std::time::Duration::from_millis(50);

/// Follows several streams with one cursor keyed `(stream, camera)`: each [`Followers::wait`] gives every new row of
/// any of them, as `(stream, camera, batch)`, batches in the order of their first row's `ts_ns`.
pub struct Followers {
    root: PathBuf,
    followers: Vec<(String, Follower)>,
    wake: Option<Wake>,
}

impl Followers {
    /// Follows `streams` for `cams` (every camera when `None`) from `cursor`.
    #[must_use]
    pub fn new(
        root: &Path,
        streams: &[&str],
        cams: Option<Vec<String>>,
        cursor: &BTreeMap<(String, String), u64>,
    ) -> Self {
        let followers = streams
            .iter()
            .map(|&stream| {
                let mine: Cursor =
                    cursor.iter().filter(|((s, _), _)| s == stream).map(|((_, c), &n)| (c.clone(), n)).collect();
                (stream.to_owned(), Follower::new(root, stream, cams.clone(), mine))
            })
            .collect();
        Self { root: root.to_path_buf(), followers, wake: None }
    }

    /// Where it is, to save and resume from.
    #[must_use]
    pub fn cursor(&self) -> BTreeMap<(String, String), u64> {
        self.followers
            .iter()
            .flat_map(|(s, f)| f.cursor().iter().map(move |(c, &n)| ((s.clone(), c.clone()), n)))
            .collect()
    }

    /// The new rows of every stream, waiting up to `timeout` when there are none.
    ///
    /// # Errors
    ///
    /// As [`Follower::poll`].
    pub fn wait(&mut self, timeout: std::time::Duration) -> Result<Vec<(String, String, RecordBatch)>> {
        // Nothing to wait for: no descriptor, no watches.
        let mut got = if timeout.is_zero() { self.poll()? } else { self.wait_for_rows(timeout)? };
        let first = |b: &RecordBatch| column::<Int64Array>(b, TS).map(|t| t.value(0)).unwrap_or(i64::MAX);
        got.sort_by_key(|(_, _, b)| first(b));
        Ok(got)
    }

    /// [`Followers::poll`] until there are rows or `timeout` passes, sleeping as [`Follower::wait`] does.
    fn wait_for_rows(&mut self, timeout: std::time::Duration) -> Result<Vec<(String, String, RecordBatch)>> {
        let deadline = deadline_after(timeout);
        let mut wake = self.wake.take().unwrap_or_else(Wake::new);
        let got = loop {
            let dirs: Vec<PathBuf> = self.followers.iter().flat_map(|(_, f)| f.dirs()).collect();
            wake.watch(&self.root, dirs.iter().map(PathBuf::as_path));
            let got = match self.poll() {
                Ok(got) => got,
                Err(e) => break Err(e),
            };
            let now = std::time::Instant::now();
            if !got.is_empty() || now >= deadline {
                break Ok(got);
            }
            wake.sleep(deadline - now);
        };
        self.wake = Some(wake);
        got
    }

    /// Every stream's new rows, without waiting.
    fn poll(&mut self) -> Result<Vec<(String, String, RecordBatch)>> {
        let mut got = Vec::new();
        for (stream, f) in &mut self.followers {
            got.extend(f.poll()?.into_iter().map(|(cam, b)| (stream.clone(), cam, b)));
        }
        Ok(got)
    }
}

/// Where each camera of `stream` ends now: a cursor that follows from here on, without reading what came before.
/// Only the open part (at most an hour) is read, to count its rows.
///
/// # Errors
///
/// [`Error::Invalid`] for a bad name, or an I/O or Arrow error.
pub fn tail(root: &Path, stream: &str, cams: Option<&[&str]>) -> Result<Cursor> {
    let mut out = Cursor::new();
    for cam in wanted(root, stream, cams)? {
        let dir = root.join(stream).join(&cam);
        let mut tries = 0;
        let next = loop {
            match camera_end(&dir) {
                // The open part closed between the listing and its read: list again.
                Err(Error::Io(_, e)) if e.kind() == std::io::ErrorKind::NotFound && tries < RELIST_TRIES => tries += 1,
                done => break done?,
            }
        };
        out.insert(cam, next);
    }
    Ok(out)
}

/// One past the last `row_seq` a camera's directory holds or ever held.
fn camera_end(dir: &Path) -> Result<u64> {
    let mut next = crate::writer::read_u64(&dir.join(".seq"))?.unwrap_or(0);
    for p in parts(dir)? {
        let end = match p.end {
            Some(end) => end,
            None => p.first + decode_file(&p.path)?.iter().map(|b| b.num_rows() as u64).sum::<u64>(),
        };
        next = next.max(end);
    }
    Ok(next)
}
