//! One-off migration (#62): rewrites a catalog's chunk locators that name files under a data root by absolute path
//! (or against an old working directory) to their form relative to the data root, so a moved install reads its
//! older chunks again (`continuum_core::chunk_path`). Idempotent; each batch is one short transaction.
//!
//! Stop the broker first (this takes its lock and refuses while the broker holds it), run, then start the broker
//! again:
//!
//! ```sh
//! continuum/target/release/continuum-relative-locators --catalog <state>/continuum/catalog.redb
//! ```

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use continuum_catalog::{Catalog, RedbCatalog};

/// Location rows per transaction: short enough that each commit is quick, long enough that the run isn't all
/// commit overhead.
const BATCH: usize = 2_000;

fn main() -> ExitCode {
    let mut catalog = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--catalog" => catalog = args.next().map(PathBuf::from),
            _ => catalog = None,
        }
    }
    let Some(catalog) = catalog else {
        eprintln!("usage: continuum-relative-locators --catalog PATH");
        return ExitCode::FAILURE;
    };
    match run(&catalog) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("continuum-relative-locators: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(path: &Path) -> Result<(), String> {
    if !path.is_file() {
        return Err(format!("{}: no catalog file there", path.display()));
    }
    let lock_path = path.with_file_name("broker.sock.lock");
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|e| format!("{}: {e}", lock_path.display()))?;
    lock.try_lock().map_err(|_| format!("the broker holds {}: stop it first", lock_path.display()))?;
    let c = RedbCatalog::open(path).map_err(|e| e.to_string())?;
    let (mut scanned, mut rewritten) = (0usize, 0usize);
    let mut after: Option<Vec<u8>> = None;
    loop {
        let batch = c.relativize_chunk_locators(after.as_deref(), BATCH).map_err(|e| e.to_string())?;
        scanned += batch.scanned;
        rewritten += batch.rewritten;
        match batch.next {
            Some(next) => after = Some(next),
            None => break,
        }
    }
    c.close(std::time::Duration::from_secs(60)).map_err(|e| e.to_string())?;
    println!("{scanned} location rows looked at, {rewritten} chunk locators rewritten relative to the data root");
    Ok(())
}
