//! The one writer of a stream's camera: appends batches to the open hour's part, starts the next hour's part, writes
//! late rows to parts of their own, and closes cleanly or is recovered by the next writer.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{Array, Int64Array, RecordBatch, UInt64Array};
use arrow_ipc::writer::{IpcWriteOptions, StreamWriter};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use crate::{Error, Part, ROW_SEQ, Result, TS, check_name, hour_of, io, parts};

/// How often an open part is synced: a power loss costs at most this much of it.
pub const SYNC_EVERY: Duration = Duration::from_secs(5);

struct Open {
    hour: u64,
    part: u32,
    first: u64,
    rows: u64,
    path: PathBuf,
    writer: StreamWriter<File>,
    synced: Instant,
    /// The deletion log's length when the part started: a forget after it may name rows already in the part.
    log_len: u64,
}

/// Appends a stream's rows for one camera. Only one writer per stream and camera, in any process, holds its lock.
pub struct Writer {
    root: PathBuf,
    stream: String,
    camera: String,
    /// The forgotten ranges covering this camera, and the deletion log's length when they were read.
    forgotten: (u64, Vec<crate::deletions::Filter>),
    dir: PathBuf,
    user: SchemaRef,
    stored: SchemaRef,
    next_seq: u64,
    /// The highest part number used for each hour, so a new part never reuses one.
    last_part: BTreeMap<u64, u32>,
    open: Option<Open>,
    /// The longest `end_ns - ts_ns` written, for overlap reads; kept in `.maxspan`.
    max_span: u64,
    _lock: File,
}

impl Writer {
    /// Takes the writer for `stream`'s `camera` under `root`. `schema` is the rows' columns: it must have a non-null
    /// `ts_ns` Int64 and no `row_seq`, which the writer adds. A part a dead writer left open is closed at the rows it
    /// completed, and `row_seq` carries on after the highest ever written.
    ///
    /// # Errors
    ///
    /// [`Error::Busy`] while another writer holds it, [`Error::Invalid`] for a bad name or schema, or an I/O error.
    pub fn open(root: &Path, stream: &str, camera: &str, schema: SchemaRef) -> Result<Self> {
        Self::open_at(root, stream, camera, schema, None)
    }

    /// [`Writer::open`], then, with `resume_at`, drops every row at or past that `row_seq`: a worker that saved its
    /// outputs' next `row_seq` with its input cursor reopens at it, replays its inputs, and writes each output once.
    /// The part that straddles it is rewritten (its version bumped) and later parts are deleted.
    ///
    /// # Errors
    ///
    /// As [`Writer::open`], and [`Error::Invalid`] for a `resume_at` past every row written.
    pub fn open_at(root: &Path, stream: &str, camera: &str, schema: SchemaRef, resume_at: Option<u64>) -> Result<Self> {
        check_name("stream", stream)?;
        check_name("camera", camera)?;
        match schema.field_with_name(TS) {
            Ok(f) if f.data_type() == &DataType::Int64 && !f.is_nullable() => {}
            _ => return Err(Error::Invalid(format!("a segment schema needs a non-null Int64 {TS}"))),
        }
        if schema.field_with_name(ROW_SEQ).is_ok() {
            return Err(Error::Invalid(format!("{ROW_SEQ} is the writer's to add")));
        }
        let dir = root.join(stream).join(camera);
        std::fs::create_dir_all(&dir).map_err(io(&dir))?;
        let lock_path = dir.join(".writer.lock");
        let lock = File::options().create(true).truncate(false).write(true).open(&lock_path).map_err(io(&lock_path))?;
        if lock.try_lock().is_err() {
            return Err(Error::Busy(dir));
        }
        crate::clear_stale(&dir)?;
        let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
        fields.push(Field::new(ROW_SEQ, DataType::UInt64, false));
        let stored = Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
        let mut next_seq = read_seq(&dir)?;
        let mut last_part = BTreeMap::new();
        for p in parts(&dir)? {
            let end = match p.end {
                Some(end) => end,
                None => {
                    let end = recover(&p)?;
                    // A forget while it was open only hid its rows; closed now, it can lose them.
                    let closed = dir.join(Part::name(p.hour, p.part, p.first, Some(end)));
                    crate::deletions::purge_closed(root, stream, camera, &closed)?;
                    end
                }
            };
            next_seq = next_seq.max(end);
            let n = last_part.entry(p.hour).or_insert(p.part);
            *n = (*n).max(p.part);
        }
        if let Some(at) = resume_at {
            if at > next_seq {
                return Err(Error::Invalid(format!("can't resume at row {at}: only {next_seq} were written")));
            }
            for p in parts(&dir)? {
                let end = p.end.unwrap_or(p.first);
                if p.first >= at {
                    std::fs::remove_file(&p.path).map_err(io(&p.path))?;
                } else if end > at {
                    rewrite_until(&p, at, &stored)?;
                }
            }
            next_seq = at;
            write_seq(&dir, next_seq)?;
        }
        let max_span = read_u64(&dir.join(".maxspan"))?.unwrap_or(0);
        Ok(Self {
            root: root.to_path_buf(),
            stream: stream.to_owned(),
            camera: camera.to_owned(),
            forgotten: (u64::MAX, Vec::new()),
            dir,
            user: schema,
            stored,
            next_seq,
            last_part,
            open: None,
            max_span,
            _lock: lock,
        })
    }

    /// The `row_seq` the next row will get.
    #[must_use]
    pub const fn next_seq(&self) -> u64 {
        self.next_seq
    }

    /// Appends `batch` (the schema's columns, in its order) and returns the `row_seq` of its first row. Rows go to
    /// their hour's part, in the batch's order; a row for an hour before the open one goes to a new part of its own.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a batch of another schema, or an Arrow or I/O error.
    pub fn append(&mut self, batch: &RecordBatch) -> Result<u64> {
        if batch.schema().fields() != self.user.fields() {
            return Err(Error::Invalid("the batch's columns aren't the writer's schema".to_owned()));
        }
        // A forgotten range stays forgotten: rows inside one are dropped here, so a late writer can't bring them back.
        let log_len = crate::deletions::log_len(&self.root);
        if log_len != self.forgotten.0 {
            self.forgotten = (log_len, crate::deletions::covering(&self.root, &self.stream, &self.camera)?);
        }
        let filtered;
        let batch = if self.forgotten.1.is_empty() {
            batch
        } else {
            filtered = crate::deletions::without(batch.clone(), &self.forgotten.1)?;
            &filtered
        };
        let first = self.next_seq;
        let ts = batch
            .column_by_name(TS)
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
            .ok_or_else(|| Error::Invalid(format!("{TS} isn't Int64")))?;
        if ts.null_count() > 0 || ts.values().iter().any(|&t| t < 0) {
            return Err(Error::Invalid(format!("{TS} must be set and not before 1970")));
        }
        if let Some(end) = batch.column_by_name(crate::END).and_then(|c| c.as_any().downcast_ref::<Int64Array>()) {
            let span = end.values().iter().zip(ts.values()).map(|(&e, &t)| e.saturating_sub(t)).max().unwrap_or(0);
            let span = u64::try_from(span).unwrap_or(0);
            if span > self.max_span {
                self.max_span = span;
                write_u64(&self.dir.join(".maxspan"), span)?;
            }
        }
        // Runs of consecutive rows in one hour, in order: a batch normally is one run.
        let mut start = 0;
        while start < batch.num_rows() {
            #[allow(clippy::cast_sign_loss)] // checked nonnegative above
            let hour = hour_of(ts.value(start) as u64);
            let mut end = start + 1;
            #[allow(clippy::cast_sign_loss)]
            while end < batch.num_rows() && hour_of(ts.value(end) as u64) == hour {
                end += 1;
            }
            self.write_run(hour, &batch.slice(start, end - start))?;
            start = end;
        }
        if let Some(open) = &mut self.open
            && open.synced.elapsed() >= SYNC_EVERY
        {
            open.writer.flush()?;
            open.writer.get_ref().sync_data().map_err(io(&open.path))?;
            open.synced = Instant::now();
        }
        Ok(first)
    }

    fn write_run(&mut self, hour: u64, rows: &RecordBatch) -> Result<()> {
        let n = rows.num_rows() as u64;
        let seq: UInt64Array = (self.next_seq..self.next_seq + n).collect();
        let mut columns = rows.columns().to_vec();
        columns.push(Arc::new(seq));
        let stored = RecordBatch::try_new(Arc::clone(&self.stored), columns)?;
        match &self.open {
            Some(open) if open.hour == hour => {}
            Some(open) if hour < open.hour => {
                // A late row: a part of its own for that hour, closed at once.
                let mut late = self.start_part(hour)?;
                late.writer.write(&stored)?;
                late.rows += n;
                self.next_seq += n;
                return self.finish(late);
            }
            _ => {
                if let Some(old) = self.open.take() {
                    self.finish(old)?;
                }
                self.open = Some(self.start_part(hour)?);
            }
        }
        let open = self.open.as_mut().expect("opened above");
        open.writer.write(&stored)?;
        open.writer.flush()?;
        open.rows += n;
        self.next_seq += n;
        Ok(())
    }

    fn start_part(&mut self, hour: u64) -> Result<Open> {
        let part = self.last_part.get(&hour).map_or(0, |p| p + 1);
        self.last_part.insert(hour, part);
        let path = self.dir.join(Part::name(hour, part, self.next_seq, None));
        let file = File::options().create_new(true).write(true).open(&path).map_err(io(&path))?;
        let options = IpcWriteOptions::default().try_with_compression(Some(arrow_ipc::CompressionType::ZSTD))?;
        let writer = StreamWriter::try_new_with_options(file, &self.stored, options)?;
        let log_len = self.forgotten.0;
        Ok(Open { hour, part, first: self.next_seq, rows: 0, path, writer, synced: Instant::now(), log_len })
    }

    /// Ends a part's stream, syncs it, and renames it closed; then records the next `row_seq`. A forget that came
    /// while the part was open only hid the rows it named there (forgets rewrite closed parts only), so a part closed
    /// after one is rewritten without them (#3210).
    fn finish(&self, mut open: Open) -> Result<()> {
        open.writer.finish()?;
        open.writer.get_ref().sync_data().map_err(io(&open.path))?;
        let closed = self.dir.join(Part::name(open.hour, open.part, open.first, Some(open.first + open.rows)));
        std::fs::rename(&open.path, &closed).map_err(io(&open.path))?;
        write_seq(&self.dir, self.next_seq)?;
        if crate::deletions::log_len(&self.root) != open.log_len {
            crate::deletions::purge_closed(&self.root, &self.stream, &self.camera, &closed)?;
        }
        Ok(())
    }

    /// Puts every row appended so far on disk and returns the next `row_seq`: the durability point a worker syncs to
    /// before it saves that `row_seq` in its checkpoint, so a crash between the two only redoes work. Appends are
    /// otherwise synced every [`SYNC_EVERY`].
    ///
    /// # Errors
    ///
    /// An Arrow or I/O error.
    pub fn sync(&mut self) -> Result<u64> {
        if let Some(open) = &mut self.open {
            open.writer.flush()?;
            open.writer.get_ref().sync_data().map_err(io(&open.path))?;
            open.synced = Instant::now();
        }
        Ok(self.next_seq)
    }

    /// Closes the open part. Dropping the writer does too, ignoring errors.
    ///
    /// # Errors
    ///
    /// An Arrow or I/O error.
    pub fn close(mut self) -> Result<()> {
        match self.open.take() {
            Some(open) => self.finish(open),
            None => Ok(()),
        }
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        if let Some(open) = self.open.take() {
            let _ = self.finish(open);
        }
    }
}

/// The next `row_seq` recorded for a camera, which outlives its parts.
fn read_seq(dir: &Path) -> Result<u64> {
    Ok(read_u64(&dir.join(".seq"))?.unwrap_or(0))
}

fn write_seq(dir: &Path, next: u64) -> Result<()> {
    write_u64(&dir.join(".seq"), next)
}

/// A small number kept in a file of its own, or `None` when there's no file.
pub(crate) fn read_u64(path: &Path) -> Result<Option<u64>> {
    match std::fs::read_to_string(path) {
        Ok(s) => s.trim().parse().map(Some).map_err(|_| Error::Invalid(format!("{} isn't a number", path.display()))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(Error::Io(path.to_path_buf(), e)),
    }
}

/// Writes a small number to its own file: synced before the rename, so a power cut leaves the old value or the new
/// one, never an empty file that [`read_u64`] refuses (which would keep the writer from opening).
fn write_u64(path: &Path, value: u64) -> Result<()> {
    use std::io::Write as _;
    let tmp = path.with_extension("tmp");
    let mut file = File::create(&tmp).map_err(io(&tmp))?;
    file.write_all(value.to_string().as_bytes()).and_then(|()| file.sync_data()).map_err(io(&tmp))?;
    std::fs::rename(&tmp, path).map_err(io(path))
}

/// Rewrites a part keeping only its rows before `at`, as the next version, and removes the old file.
fn rewrite_until(p: &Part, at: u64, stored: &SchemaRef) -> Result<()> {
    let dir = p.path.parent().unwrap_or_else(|| Path::new("."));
    let path = dir.join(Part::versioned_name(p.hour, p.part, p.first, at, p.version + 1));
    let mut kept = Vec::new();
    for batch in crate::reader::decode_file(&p.path)? {
        let seq = batch
            .column_by_name(ROW_SEQ)
            .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
            .ok_or_else(|| Error::Invalid(format!("{} has no {ROW_SEQ}", p.path.display())))?;
        let keep: arrow_array::BooleanArray = seq.values().iter().map(|&s| Some(s < at)).collect();
        kept.push(arrow_select::filter::filter_record_batch(&batch, &keep)?);
    }
    crate::replace_part(&path, &p.path, stored, &kept)
}

/// Closes a part a dead writer left open, at the rows it completed (a torn last batch is left where it is, after
/// them, and readers stop before it). Returns one past its last complete row's `row_seq`.
fn recover(p: &Part) -> Result<u64> {
    let mut rows = 0u64;
    for batch in crate::reader::decode_file(&p.path)? {
        rows += batch.num_rows() as u64;
    }
    let end = p.first + rows;
    let dir = p.path.parent().unwrap_or_else(|| Path::new("."));
    let closed = dir.join(Part::name(p.hour, p.part, p.first, Some(end)));
    std::fs::rename(&p.path, &closed).map_err(io(&p.path))?;
    Ok(end)
}
