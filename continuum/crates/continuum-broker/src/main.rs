//! `continuum-broker`: runs the broker on a Unix socket until SIGTERM or Ctrl-C.
//!
//! ```sh
//! continuum-broker --state-dir data/state          # socket at data/state/continuum/broker.sock
//! continuum-broker --socket /tmp/cnt/broker.sock   # an explicit socket path
//! RUST_LOG=continuum_broker=debug continuum-broker  # more logging
//! ```

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::Parser;
use continuum_broker::server::{Config, Server};
use tokio::signal::unix::{SignalKind, signal};
use tracing::error;
use tracing::{info, warn};

mod capture_sizing;

/// Continuum Broker daemon.
#[derive(Debug, Parser)]
#[command(version)]
struct Args {
    /// The pack's state directory (paths.state); the socket goes in its continuum/ folder.
    #[arg(long, default_value = "data/state")]
    state_dir: PathBuf,
    /// The socket path, instead of the one under --state-dir.
    #[arg(long)]
    socket: Option<PathBuf>,
    /// Close a session that sends nothing, or doesn't read its replies, for this many seconds.
    #[arg(long, default_value_t = 30)]
    heartbeat_timeout_s: u64,
    /// The most sessions open at once; more connections are refused.
    #[arg(long, default_value_t = continuum_broker::server::DEFAULT_MAX_SESSIONS)]
    max_sessions: usize,
    /// How long one step of a long catalog job (a purge, a scrub) should take, in milliseconds. Ingest waits
    /// behind at most one step, so keep it small.
    #[arg(long, default_value_t = 5)]
    step_target_ms: u64,
    /// The machine profile the scheduler budgets for: mac_mini_64gb or linux_dual_3090 (default: this machine's).
    #[arg(long)]
    profile: Option<String>,
    /// The Tier 0 arena's slot classes as SIZExCOUNT, comma separated (like 16384x64,2097152x8), instead of the
    /// profile's. For tests and small machines: the profile's arena is gigabytes.
    #[arg(long)]
    tier0_classes: Option<String>,
    /// Capture's device manifest: reserve additional camera slots from its effective video settings.
    #[arg(long)]
    manifest: Option<PathBuf>,
    /// Maximum additional camera reservation in MiB; explicit retention that exceeds this is rejected.
    #[arg(long, default_value_t = capture_sizing::DEFAULT_CAMERA_BYTES >> 20)]
    tier0_camera_mib: u64,
    /// Print the effective Tier 0 reservation and exit without creating a socket, catalog, or shared memory.
    #[arg(long)]
    print_tier0_plan: bool,
    /// The redb catalog file, instead of catalog.redb beside the socket.
    #[arg(long)]
    catalog: Option<PathBuf>,
    /// Never scrub the catalog file into a fresh one (which gives its freed pages back to the disk and leaves no
    /// purged bytes in them). On by default; see the scrub module for when it runs.
    #[arg(long)]
    no_catalog_scrub: bool,
    /// MiB reserved in the Tier 0 arena for rings carved when streams register (PCM, IMU).
    #[arg(long, default_value_t = continuum_broker::server::DEFAULT_RING_BYTES >> 20)]
    tier0_ring_mib: u64,
}

fn parse_classes(s: &str) -> Result<Vec<continuum_sched::SlotClass>, String> {
    s.split(',')
        .map(|c| {
            let (size, count) = c.split_once('x').ok_or_else(|| format!("{c:?} isn't SIZExCOUNT"))?;
            let size_bytes = size.trim().parse().map_err(|e| format!("{size:?}: {e}"))?;
            let count = count.trim().parse().map_err(|e| format!("{count:?}: {e}"))?;
            Ok(continuum_sched::SlotClass { size_bytes, count })
        })
        .collect()
}

fn capture_classes(
    args: &Args,
    profile: &str,
    ring_bytes: u64,
) -> Result<Option<Vec<continuum_sched::SlotClass>>, String> {
    if let Some(explicit) = &args.tier0_classes {
        let classes = parse_classes(explicit)?;
        if args.print_tier0_plan {
            let plan = capture_sizing::plan("", &classes, false, 0, ring_bytes)?;
            print_plan("explicit --tier0-classes", &plan, ring_bytes);
        }
        return Ok(Some(classes)); // Overrides deliberately bypass manifest sizing and its budget.
    }
    if args.manifest.is_none() && !args.print_tier0_plan {
        return Ok(None);
    }
    let baseline =
        continuum_sched::Profile::named(profile).ok_or_else(|| format!("no machine profile named {profile}"))?;
    let Some(path) = &args.manifest else {
        let plan = capture_sizing::plan("", &baseline.slot_classes, false, 0, ring_bytes)?;
        print_plan("profile (no --manifest)", &plan, ring_bytes);
        return Ok(None);
    };
    let budget = args.tier0_camera_mib.checked_mul(1 << 20).ok_or("--tier0-camera-mib overflows bytes")?;
    match capture_sizing::load(path, &baseline.slot_classes, cfg!(target_os = "macos"), budget, ring_bytes)? {
        Some(plan) => {
            info!(manifest = %path.display(), cameras = plan.cameras.len(), camera_bytes = plan.camera_bytes,
                total_arena_bytes = plan.total_bytes, ring_bytes, "sized Tier 0 from capture settings");
            if args.print_tier0_plan {
                print_plan(&path.display().to_string(), &plan, ring_bytes);
            }
            Ok(Some(plan.classes))
        }
        None => {
            warn!(manifest = %path.display(), "capture manifest is missing; using profile Tier 0 classes until device setup and broker restart");
            if args.print_tier0_plan {
                let plan = capture_sizing::plan("", &baseline.slot_classes, false, 0, ring_bytes)?;
                print_plan("profile (missing manifest)", &plan, ring_bytes);
            }
            Ok(None)
        }
    }
}

fn print_plan(source: &str, plan: &capture_sizing::Plan, ring_bytes: u64) {
    println!("tier0 source={source}");
    println!(
        "baseline_bytes={} camera_increment_bytes={} ring_region_bytes={} total_arena_bytes={}",
        plan.baseline_bytes, plan.camera_bytes, ring_bytes, plan.total_bytes
    );
    for camera in &plan.cameras {
        if let Some(classes) = &camera.network_class_slots {
            println!(
                "camera={} format={} nominal_fps={} target_ring_seconds={} byte_budget_limited=true class_slots={classes:?}",
                camera.name, camera.format, camera.fps, camera.ring_seconds
            );
        } else {
            println!(
                "camera={} format={} fps={} ring_seconds={} slots_per_class={} class_sizes={:?}",
                camera.name, camera.format, camera.fps, camera.ring_seconds, camera.slots_per_class, camera.class_sizes
            );
        }
    }
    for (index, class) in plan.classes.iter().enumerate() {
        println!("class={index} configured_slot_bytes={} count={}", class.size_bytes, class.count);
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    if std::env::args().any(|arg| arg == "--capabilities") {
        println!("{}", serde_json::json!({"version": 1, "rtsp_h264": true}));
        return ExitCode::SUCCESS;
    }
    continuum_broker::log::init();
    let args = Args::parse();
    let profile = args.profile.clone().unwrap_or_else(|| continuum_broker::server::default_profile().to_owned());
    let Some(ring_bytes) = args.tier0_ring_mib.checked_mul(1 << 20) else {
        error!("--tier0-ring-mib overflows bytes");
        return ExitCode::FAILURE;
    };
    let tier0 = match capture_classes(&args, &profile, ring_bytes) {
        Ok(classes) => classes,
        Err(error) => {
            error!("Tier 0 configuration: {error}");
            return ExitCode::FAILURE;
        }
    };
    if args.print_tier0_plan {
        return ExitCode::SUCCESS;
    }
    let socket = args.socket.unwrap_or_else(|| args.state_dir.join("continuum").join("broker.sock"));
    // The catalog lives beside the socket unless it's put elsewhere.
    let catalog = args.catalog.unwrap_or_else(|| socket.with_file_name("catalog.redb"));
    let config = Config {
        socket,
        catalog: Some(catalog),
        heartbeat_timeout: Duration::from_secs(args.heartbeat_timeout_s.max(1)),
        max_sessions: args.max_sessions.max(1),
        writer: continuum_broker::core::WriterConfig {
            step_target: Duration::from_millis(args.step_target_ms.max(1)),
            ..Default::default()
        },
        profile,
        tier0_ring_bytes: ring_bytes,
        tier0,
        forget: continuum_broker::core::ForgetConfig::default(),
        scrub: continuum_broker::scrub::ScrubConfig { enabled: !args.no_catalog_scrub, ..Default::default() },
    };
    // Watched before binding: the socket accepts during bind, and a SIGTERM before this would skip the clean stop
    // (the arena and socket left behind).
    let mut term = match signal(SignalKind::terminate()) {
        Ok(s) => s,
        Err(e) => {
            error!("couldn't watch for SIGTERM: {e}");
            return ExitCode::FAILURE;
        }
    };
    let server = match Server::bind(config).await {
        Ok(s) => s,
        Err(e) => {
            error!("{e}");
            return ExitCode::FAILURE;
        }
    };
    server
        .run_until(async move {
            tokio::select! {
                _ = term.recv() => {}
                _ = tokio::signal::ctrl_c() => {}
            }
        })
        .await;
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_classes_still_bypass_manifest_and_camera_budget() {
        let args = Args::parse_from([
            "broker",
            "--tier0-classes",
            "16384x4",
            "--manifest",
            "absent.toml",
            "--tier0-camera-mib",
            "0",
        ]);
        assert_eq!(
            capture_classes(&args, "not-a-profile", 0).unwrap(),
            Some(vec![continuum_sched::SlotClass { size_bytes: 16384, count: 4 }])
        );
    }

    #[test]
    fn invalid_explicit_classes_are_not_replaced_with_defaults() {
        let args = Args::parse_from(["broker", "--tier0-classes", "broken"]);
        assert!(capture_classes(&args, "mac_mini_64gb", 0).is_err());
    }
}
