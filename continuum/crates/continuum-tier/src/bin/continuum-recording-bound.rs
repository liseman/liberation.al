//! One-time repair, offline: give each instant audio original that holds its stream's recording history unknown the
//! end its own samples bound. Run it from the repository root with the broker stopped, as
//! `continuum-recording-bound --catalog data/state/continuum/catalog.redb [--apply]`. It opens the catalog file
//! directly, so it first takes the broker's own lock (`broker.sock.lock` beside the catalog) without waiting: it
//! refuses while the broker runs, and a broker can't start while it works. Without `--apply` it only reports.
//!
//! An original is bounded only when its chunk frame is found, that frame's record absorbed exactly this original at
//! its own start, and the record is a CPB2 batch whose true source rate agrees with the stream's. The end is its
//! start plus its frames over that rate, widened by capture's drift bound (1000 ppm) and one block. The row stays
//! unknown over that extent; anything missing or disagreeing leaves it as it is.

use std::path::PathBuf;
use std::process::ExitCode;

use continuum_catalog::{Catalog, RedbCatalog};
use continuum_tier::recording_bound::bounded_end;

fn main() -> ExitCode {
    let mut catalog = None;
    let mut apply = false;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--catalog" => catalog = args.next().map(PathBuf::from),
            "--apply" => apply = true,
            _ => catalog = None,
        }
    }
    let Some(catalog) = catalog else {
        eprintln!("usage: continuum-recording-bound --catalog PATH [--apply]");
        return ExitCode::FAILURE;
    };
    match run(&catalog, apply) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("continuum-recording-bound: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(path: &std::path::Path, apply: bool) -> Result<(), String> {
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
    let streams = c.streams(None, 100_000).map_err(|e| e.to_string())?;
    let originals = c.instant_originals().map_err(|e| e.to_string())?;
    if originals.is_empty() {
        println!("no instant audio original holds a stream");
    }
    for original in originals {
        let label = format!("stream {} original {} at {} ns", original.stream.0, original.original, original.start_ns);
        let rate = streams.iter().find(|(id, _)| *id == original.stream).and_then(|(_, info)| info.rate_hz);
        match bounded_end(&c, &original, rate) {
            Ok(end_ns) => {
                let ms = (end_ns - original.start_ns) as f64 / 1e6;
                if apply {
                    let known = c.bound_instant_original(original.original, end_ns).map_err(|e| e.to_string())?;
                    println!("{label}: bounded to {end_ns} ns (+{ms:.1} ms); stream history known: {known}");
                } else {
                    println!("{label}: would bound to {end_ns} ns (+{ms:.1} ms); run with --apply to write");
                }
            }
            Err(why) => println!("{label}: left held: {why}"),
        }
    }
    drop(c);
    drop(lock);
    Ok(())
}
