//! Chunk files no catalog row names: the leftovers of the catalog's crash window.
//!
//! The catalog fsyncs about once a second, so a broker crash or power loss can roll back a chunk commit the
//! broker had already acknowledged. The tier worker's saved positions roll back with it and the frames still in
//! Tier 0 are absorbed again, so nothing is lost but the first chunk file, which no catalog row names any more.
//!
//! Such a file can only be one the previous run wrote shortly before this run started. So an hour after it starts
//! ([`AFTER`], when no commit of the previous run can still be in flight and every recovered chunk has had its
//! chance to commit), the worker looks once at the finished raw chunks named as created in the [`LOOKBACK_NS`] before
//! its start, skipping the ones startup recovery handed back. Per stream, one locator query over their span says
//! which files the catalog names; the rest are deleted, or with `CONTINUUM_TIER_SWEEP_DRY_RUN` set only counted. The
//! counts are logged. Copies (`-copy-`) and set-aside files are never touched.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

use continuum_core::StreamId;

use crate::TierError;
use crate::chunk::{ChunkReader, Record};

/// How long after the worker starts it sweeps.
pub const AFTER: Duration = Duration::from_secs(3600);
/// How far before the start a chunk may have been created and still be looked at: a commit lost in the crash window
/// is at most a few chunks old, and this keeps the look to a few thousand files rather than the whole directory.
pub const LOOKBACK_NS: u64 = 10 * 60 * 1_000_000_000;
/// Set (to anything) to count unnamed files without deleting them.
pub const DRY_RUN_ENV: &str = "CONTINUUM_TIER_SWEEP_DRY_RUN";

/// A chunk file the sweep looks at: its stream, path, and the span of its records.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    /// Its stream, from its name.
    pub stream: StreamId,
    /// The file.
    pub path: PathBuf,
    /// Its records' span, UTC nanoseconds, end exclusive.
    pub span: (u64, u64),
}

/// What a sweep did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// Files looked at.
    pub checked: usize,
    /// Of those, the ones no catalog row names.
    pub unnamed: usize,
    /// Of those, the ones deleted (none in a dry run).
    pub deleted: usize,
    /// Files passed over: a stream with no name yet, or a file that couldn't be read or removed.
    pub skipped: usize,
}

/// The finished raw chunks in `dir` named as created in `[before_ns - LOOKBACK_NS, before_ns)`, but none of `skip`.
/// A file that isn't a whole chunk with records is passed over (counted in `skipped`).
///
/// # Errors
///
/// An I/O error listing the directory.
pub fn candidates(
    dir: &Path,
    before_ns: u64,
    skip: &BTreeSet<PathBuf>,
    skipped: &mut usize,
) -> Result<Vec<Candidate>, TierError> {
    let mut out = Vec::new();
    if !dir.exists() {
        return Ok(out);
    }
    let since = before_ns.saturating_sub(LOOKBACK_NS);
    for path in crate::layout::chunk_paths(dir, None)? {
        let Some((stream, created_ns)) = raw_chunk_name(&path) else { continue };
        if !(since..before_ns).contains(&created_ns) || skip.contains(&path) {
            continue;
        }
        let span = ChunkReader::open(&path).ok().and_then(|r| span(r.records()));
        match span {
            Some(span) => out.push(Candidate { stream, path, span }),
            None => *skipped += 1,
        }
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

/// `<stream>-<created ns>-<counter>.cnk`, as the flusher names a raw chunk: its stream and creation time. None for
/// anything else (a copy, a set-aside file, a stray).
fn raw_chunk_name(path: &Path) -> Option<(StreamId, u64)> {
    if path.extension()? != "cnk" {
        return None;
    }
    let stem = path.file_stem()?.to_str()?;
    let mut parts = stem.split('-');
    let (stream, created, counter) = (parts.next()?, parts.next()?, parts.next()?);
    if parts.next().is_some() || counter.parse::<u64>().is_err() {
        return None;
    }
    Some((StreamId(stream.parse().ok()?), created.parse().ok()?))
}

fn span(records: &[Record]) -> Option<(u64, u64)> {
    let start = records.iter().map(|r| r.start_ns).min()?;
    let end = records.iter().map(Record::end_exclusive).max()?;
    Some((start, end))
}

/// The candidates grouped by stream, with each group's span hull: what one locator query per stream covers.
#[must_use]
pub fn by_stream(found: &[Candidate]) -> BTreeMap<StreamId, ((u64, u64), Vec<&Candidate>)> {
    let mut out: BTreeMap<StreamId, ((u64, u64), Vec<&Candidate>)> = BTreeMap::new();
    for c in found {
        let (hull, files) = out.entry(c.stream).or_insert(((u64::MAX, 0), Vec::new()));
        *hull = (hull.0.min(c.span.0), hull.1.max(c.span.1));
        files.push(c);
    }
    out
}

/// The files the catalog's chunk locators `uris` name, as the keys [`sweep`] compares by: each resolved against the
/// data root `root` (a relative locator is under it) and normalized, so a relative locator and the absolute path the
/// chunk directory lists for the same file are one key (`continuum_core::chunk_path`).
#[must_use]
pub fn named_files<'a>(root: &Path, uris: impl IntoIterator<Item = &'a str>) -> BTreeSet<String> {
    uris.into_iter().map(|uri| continuum_core::chunk_path::resolve(root, uri).to_string_lossy().into_owned()).collect()
}

/// Deletes (or, with `dry_run`, only counts) each of `files` whose path isn't among `named` (the catalog's chunk
/// locators over their span, as [`named_files`] keys them), adding to `report`. A candidate's path is normalized the
/// same way before it is looked up, so a chunk directory spelled with `.` or `..` still matches.
pub fn sweep(files: &[&Candidate], named: &BTreeSet<String>, dry_run: bool, report: &mut Report) {
    for c in files {
        report.checked += 1;
        let key = continuum_core::chunk_path::resolve(Path::new(""), &c.path.to_string_lossy());
        if named.contains(key.to_string_lossy().as_ref()) {
            continue;
        }
        report.unnamed += 1;
        if dry_run {
            continue;
        }
        match std::fs::remove_file(&c.path) {
            Ok(()) => {
                report.deleted += 1;
                crate::layout::prune_hour_dir(&c.path);
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                tracing::warn!("sweep: couldn't remove {}: {e}", c.path.display());
                report.skipped += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{ChunkFormat, ChunkWriter};

    const START: u64 = 1_800_000_000_000_000_000;
    const MIN: u64 = 60_000_000_000;

    /// Where a file of `stream` created at `created_ns` goes: its hour directory (`layout`), made.
    fn hour(dir: &Path, stream: u32, created_ns: u64) -> PathBuf {
        let hour = crate::layout::hour_dir(dir, StreamId(stream), created_ns);
        crate::layout::make_hour_dir(dir, &hour).unwrap();
        hour
    }

    /// A finished raw chunk of `stream` created at `created_ns`, with one record at that time.
    fn chunk(dir: &Path, stream: u32, created_ns: u64, n: u32) -> PathBuf {
        let path = hour(dir, stream, created_ns).join(format!("{stream}-{created_ns}-{n}.cnk"));
        let mut w = ChunkWriter::create(&path, StreamId(stream), ChunkFormat::Passthrough, created_ns).unwrap();
        w.append(created_ns, created_ns + 1_000, 0, b"frame").unwrap();
        w.close().unwrap();
        path
    }

    #[test]
    fn only_raw_chunks_made_just_before_the_start_and_not_recovered_are_looked_at() {
        let dir = tempfile::tempdir().unwrap();
        let recent = chunk(dir.path(), 6, START - MIN, 1);
        let recovered = chunk(dir.path(), 6, START - 2 * MIN, 2);
        chunk(dir.path(), 6, START - 11 * MIN, 3); // before the lookback
        chunk(dir.path(), 6, START + 1, 4); // this run's own
        let six = hour(dir.path(), 6, START - MIN);
        std::fs::write(six.join(format!("6-copy-{}.cnk", "x")), b"copy").unwrap();
        std::fs::write(six.join(format!("6-{}-5.rejected", START - MIN)), b"set aside").unwrap();
        let bad = hour(dir.path(), 7, START - MIN).join(format!("7-{}-6.cnk", START - MIN));
        std::fs::write(&bad, b"not a chunk").unwrap();
        let mut skipped = 0;
        let found = candidates(dir.path(), START, &BTreeSet::from([recovered]), &mut skipped).unwrap();
        assert_eq!(found, [Candidate { stream: StreamId(6), path: recent, span: (START - MIN, START - MIN + 1_000) }]);
        assert_eq!(skipped, 1, "the unreadable one");
    }

    #[test]
    fn a_file_the_catalog_names_stays_and_one_it_does_not_goes_unless_dry_run() {
        let dir = tempfile::tempdir().unwrap();
        let named = chunk(dir.path(), 6, START - 3 * MIN, 1);
        let orphan = chunk(dir.path(), 6, START - 2 * MIN, 2);
        let other = chunk(dir.path(), 8, START - MIN, 3);
        let found = candidates(dir.path(), START, &BTreeSet::new(), &mut 0).unwrap();
        let groups = by_stream(&found);
        assert_eq!(groups[&StreamId(6)].0, (START - 3 * MIN, START - 2 * MIN + 1_000), "one query's span per stream");
        let locators = BTreeSet::from([named.to_string_lossy().into_owned()]);

        let mut dry = Report::default();
        for (_, files) in groups.values() {
            sweep(files, &locators, true, &mut dry);
        }
        assert_eq!(dry, Report { checked: 3, unnamed: 2, deleted: 0, skipped: 0 });
        assert!(orphan.exists() && other.exists(), "a dry run deletes nothing");

        let mut report = Report::default();
        for (_, files) in groups.values() {
            sweep(files, &locators, false, &mut report);
        }
        assert_eq!(report, Report { checked: 3, unnamed: 2, deleted: 2, skipped: 0 });
        assert!(named.exists() && !orphan.exists() && !other.exists());
    }

    #[test]
    fn a_file_named_by_a_relative_locator_stays_however_the_chunk_directory_is_spelled() {
        // The catalog stores locators relative to the data root (#62), and the chunk directory lists absolute paths:
        // the sweep must take them for the same file, or it deletes every chunk of the lookback an hour after a start.
        let root = tempfile::tempdir().unwrap();
        let tier = root.path().join("continuum").join("tier");
        let named = chunk(&tier, 6, START - 3 * MIN, 1);
        let orphan = chunk(&tier, 6, START - 2 * MIN, 2);
        let relative = named.strip_prefix(root.path()).unwrap().to_string_lossy().into_owned();
        assert!(relative.starts_with("continuum/tier/"), "{relative}");
        std::fs::create_dir(root.path().join("continuum").join("x")).unwrap();
        let spelled = root.path().join("continuum").join("x").join("..").join("tier");
        let found = candidates(&spelled, START, &BTreeSet::new(), &mut 0).unwrap();
        let files: Vec<&Candidate> = found.iter().collect();
        let mut report = Report::default();
        sweep(&files, &named_files(root.path(), [relative.as_str()]), false, &mut report);
        assert_eq!(report, Report { checked: 2, unnamed: 1, deleted: 1, skipped: 0 });
        assert!(named.exists() && !orphan.exists());
    }
}
