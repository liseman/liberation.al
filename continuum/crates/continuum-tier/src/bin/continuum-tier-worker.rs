//! The tier worker (docs/CONTINUUM-STORAGE.md): follows the capture streams, drains each payload from Tier 0
//! into Tier 1 chunk files, and commits each chunk through the broker (see `continuum_tier::service`).
//!
//! ```sh
//! continuum-tier-worker --state-dir data/state                    # streams from config/devices.toml
//! continuum-tier-worker --state-dir data/state --stream cam0.video
//! ```
//!
//! With `--state-dir D`, the broker's socket is `D/continuum/broker.sock` and chunks go in `D/continuum/tier`, where
//! the gateway reads them. A microphone's chunk becomes Opus as it closes, before it is committed.
//!
//! A run lost to the broker (a cascade restarts it, and capture carves its rings after this worker subscribed) is
//! followed by a new run in the same process (`service::rides_through`), after a short wait that grows while
//! the broker stays away (`service::Backoff`). Only a chunk file problem, or a stop, ends the program.
//!
//! The opt-in AV1 archive pass (`continuum_tier::archive`) is configured by environment, so an older binary
//! simply ignores it: `CONTINUUM_AV1_ARCHIVE=True` turns it on (services/services.toml sets it from
//! `video_archive.av1`).

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use continuum_tier::FlushSettings;
use continuum_tier::service::{Backoff, Service, Settings, manifest_streams, rides_through};

const USAGE: &str = "usage: continuum-tier-worker (--state-dir D | --socket S --chunks DIR) [--manifest config/devices.toml] [--stream NAME]...";

fn main() -> ExitCode {
    continuum_log::init("continuum-tier-worker");
    if std::env::args_os().skip(1).any(|arg| arg == "--capabilities") {
        println!(
            r#"{{"version":1,"rtsp_h264":true,"av1_archive":{}}}"#,
            continuum_tier::probe::av1_encoder_available()
        );
        return ExitCode::SUCCESS;
    }
    let Ok(Args { state_dir, socket, chunks, manifest, mut streams }) = parse_args(std::env::args().skip(1)) else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    // Locators of the files under the state directory are stored relative to it (continuum_core::chunk_path).
    let data_root = state_dir.as_ref().map(|d| std::path::absolute(d).unwrap_or_else(|_| d.clone()));
    let base = state_dir.map(|d| d.join("continuum"));
    let (Some(socket), Some(chunk_dir)) = (
        socket.or_else(|| base.as_ref().map(|b| b.join("broker.sock"))),
        chunks.or_else(|| base.as_ref().map(|b| b.join("tier"))),
    ) else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    // Without a state directory, locators are these paths, and readers in other directories open them: absolute.
    let chunk_dir = std::path::absolute(&chunk_dir).unwrap_or(chunk_dir);
    let manifest = manifest.unwrap_or_else(|| PathBuf::from("config/devices.toml"));
    match std::fs::read_to_string(&manifest) {
        Ok(text) => match manifest_streams(&text) {
            Ok(found) => streams.extend(found),
            Err(e) => {
                tracing::error!("{}: {e}", manifest.display());
                return ExitCode::from(2);
            }
        },
        Err(e) if streams.is_empty() => {
            tracing::error!("no streams to follow ({}: {e}); give --stream", manifest.display());
            return ExitCode::from(2);
        }
        Err(_) => {}
    }
    streams.sort();
    streams.dedup();
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
            let mut term = match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(t) => t,
                Err(_) => return,
            };
            tokio::select! {
                _ = term.recv() => {}
                _ = tokio::signal::ctrl_c() => {}
            }
            stop.store(true, Ordering::SeqCst);
        });
    }
    tracing::info!("following {streams:?}, chunks in {}", chunk_dir.display());
    let settings = Settings {
        socket,
        chunk_dir,
        data_root,
        streams,
        flush: FlushSettings::default(),
        tick: Duration::from_millis(50),
    };
    let archive = match continuum_tier::archive::ArchiveSettings::from_env(|key| std::env::var(key).ok()) {
        Ok(archive) => archive,
        // Recording never waits on an optional pass: a bad setting is said, and the pass stays off.
        Err(e) => {
            tracing::warn!("AV1 archive off: {e}");
            None
        }
    };
    let mut backoff = Backoff::new();
    loop {
        let began = Instant::now();
        let ended = match Service::start(settings.clone(), rt.handle()) {
            Ok(mut service) => {
                if let Some(archive) = archive {
                    service.with_archive(archive);
                }
                match service.run(&stop) {
                    Ok(()) => {
                        tracing::info!("stopped: {:?}", service.counts());
                        return ExitCode::SUCCESS;
                    }
                    Err(e) => e,
                }
            }
            Err(e) => e,
        };
        if !rides_through(&ended) {
            tracing::error!("{ended}");
            return ExitCode::FAILURE;
        }
        let wait = backoff.after(began.elapsed());
        tracing::warn!("{ended}; a new run in {:.2} s", wait.as_secs_f64());
        if !wait_unless_stopped(&stop, wait) {
            tracing::info!("stopped between runs");
            return ExitCode::SUCCESS;
        }
    }
}

/// The flags after the program name.
#[derive(Default)]
struct Args {
    state_dir: Option<PathBuf>,
    socket: Option<PathBuf>,
    chunks: Option<PathBuf>,
    manifest: Option<PathBuf>,
    streams: Vec<String>,
}

/// The flags after the program name: each known one takes the argument after it as its value. An unknown flag, or a
/// flag with no value, is the usage error, and an unknown flag does not eat the argument after it.
fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Args, ()> {
    let mut out = Args::default();
    while let Some(flag) = args.next() {
        let slot = match flag.as_str() {
            "--state-dir" => &mut out.state_dir,
            "--socket" => &mut out.socket,
            "--chunks" => &mut out.chunks,
            "--manifest" => &mut out.manifest,
            "--stream" => {
                out.streams.push(args.next().ok_or(())?);
                continue;
            }
            _ => return Err(()),
        };
        *slot = Some(args.next().ok_or(())?.into());
    }
    Ok(out)
}

/// Waits `wait`, or less when `stop` is set; whether it ran the whole wait without a stop.
fn wait_unless_stopped(stop: &AtomicBool, wait: Duration) -> bool {
    let until = Instant::now() + wait;
    while Instant::now() < until {
        if stop.load(Ordering::SeqCst) {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50).min(until.saturating_duration_since(Instant::now())));
    }
    !stop.load(Ordering::SeqCst)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(lines: &[&str]) -> Result<Args, ()> {
        parse_args(lines.iter().map(|s| (*s).to_owned()))
    }

    #[test]
    fn each_flag_takes_the_argument_after_it() {
        let args = parse(&["--state-dir", "s", "--socket", "b.sock", "--chunks", "c", "--manifest", "m"]).unwrap();
        assert_eq!(args.state_dir, Some(PathBuf::from("s")));
        assert_eq!(args.socket, Some(PathBuf::from("b.sock")));
        assert_eq!(args.chunks, Some(PathBuf::from("c")));
        assert_eq!(args.manifest, Some(PathBuf::from("m")));
        assert!(args.streams.is_empty());
    }

    #[test]
    fn streams_accumulate_in_order() {
        let args = parse(&["--stream", "cam0.video", "--stream", "mic0.audio"]).unwrap();
        assert_eq!(args.streams, vec!["cam0.video".to_owned(), "mic0.audio".to_owned()]);
    }

    #[test]
    fn an_unknown_flag_is_a_usage_error() {
        assert!(parse(&["--nope"]).is_err());
        // The flag after an unknown one is still parsed, not swallowed as its value.
        assert!(parse(&["--nope", "--socket", "b.sock"]).is_err());
    }

    #[test]
    fn a_flag_with_no_value_is_a_usage_error() {
        assert!(parse(&["--socket"]).is_err());
        assert!(parse(&["--state-dir", "s", "--stream"]).is_err());
    }
}
