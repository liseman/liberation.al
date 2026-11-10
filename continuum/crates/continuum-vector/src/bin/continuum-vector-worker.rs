//! The vector worker (docs/CONTINUUM.md, "Vector worker"): answers similarity and text searches, files embeddings
//! from the embedding streams, and carries out purges on its rows (see `continuum_vector::service`).
//!
//! ```sh
//! continuum-vector-worker --state-dir data/state   # face.embedding, journal.embedding.*, journal.entry text, and
//!                                                  # voice.embedding in the segment store
//! continuum-vector-worker --state-dir data/state --embeddings face.embedding
//! ```
//!
//! With `--state-dir D`, the broker's socket is `D/continuum/broker.sock`, the store is `D/continuum/vector`, and the
//! segment store is `D/segments` (`--segments DIR` names another).
//! It exits when the broker connection ends; the service supervisor restarts it.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use continuum_vector::service::{Settings, run};

const USAGE: &str = "usage: continuum-vector-worker (--state-dir D | --socket S --store DIR) [--segments DIR] \
                     [--embeddings STREAM]... [--embedding-prefix PREFIX]... [--texts STREAM]...";

fn main() -> ExitCode {
    continuum_log::init("continuum-vector-worker");
    let (mut state_dir, mut socket, mut store, mut embeddings) =
        (None::<PathBuf>, None::<PathBuf>, None::<PathBuf>, Vec::<String>::new());
    let (mut prefixes, mut texts, mut segments) = (Vec::<String>::new(), Vec::<String>::new(), None::<PathBuf>);
    if std::env::args().skip(1).any(|arg| arg == "--version") {
        println!("continuum-vector-worker {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match (flag.as_str(), args.next()) {
            ("--state-dir", Some(v)) => state_dir = Some(v.into()),
            ("--socket", Some(v)) => socket = Some(v.into()),
            ("--store", Some(v)) => store = Some(v.into()),
            ("--embeddings", Some(v)) => embeddings.push(v),
            ("--embedding-prefix", Some(v)) => prefixes.push(v),
            ("--texts", Some(v)) => texts.push(v),
            ("--segments", Some(v)) => segments = Some(v.into()),
            _ => {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            }
        }
    }
    let segments = segments.or_else(|| state_dir.as_ref().map(|d| d.join("segments")));
    let base = state_dir.map(|d| d.join("continuum"));
    let (Some(socket), Some(store)) = (
        socket.or_else(|| base.as_ref().map(|b| b.join("broker.sock"))),
        store.or_else(|| base.as_ref().map(|b| b.join("vector"))),
    ) else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let mut settings = Settings::new(socket, store);
    settings.segments = segments;
    if !embeddings.is_empty() {
        embeddings.sort();
        embeddings.dedup();
        settings.embeddings = embeddings;
    }
    if !prefixes.is_empty() {
        settings.embedding_prefixes = prefixes;
    }
    if !texts.is_empty() {
        settings.texts = texts;
    }
    let rt = match tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            tracing::error!("no runtime: {e}");
            return ExitCode::FAILURE;
        }
    };
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = Arc::clone(&stop);
        rt.spawn(async move {
            let Ok(mut term) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) else {
                return;
            };
            tokio::select! {
                _ = term.recv() => {}
                _ = tokio::signal::ctrl_c() => {}
            }
            stop.store(true, Ordering::SeqCst);
        });
    }
    match rt.block_on(run(settings, stop)) {
        Ok(()) => {
            tracing::info!("stopped");
            ExitCode::SUCCESS
        }
        Err(e) => {
            tracing::error!("{e}");
            ExitCode::FAILURE
        }
    }
}
