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
    /// The machine profile the scheduler budgets for: mac_mini_64gb or linux_dual_3090 (default: this machine's,
    /// with its Tier 0 headroom fitted to /dev/shm; a profile named here is used whole).
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
    /// MiB reserved in the Tier 0 arena for rings carved when streams register (PCM, IMU). Default 64, or less when
    /// nothing captures and /dev/shm is small.
    #[arg(long)]
    tier0_ring_mib: Option<u64>,
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

/// The Tier 0 layout to create: the slot classes (None: the profile's own) and the ring region's bytes.
type Layout = (Option<Vec<continuum_sched::SlotClass>>, u64);

/// Chooses the arena's layout. `--tier0-classes` is used as given. Otherwise the plan is the profile's headroom
/// plus what the manifest's devices ask for; when the profile is this machine's default (no `--profile`), the
/// headroom is fitted to `/dev/shm` (`capture_sizing::fit`), so a machine or a container with little shared memory
/// still starts. A named `--profile` and an explicit `--tier0-ring-mib` are never shrunk.
fn capture_layout(args: &Args, profile: &str, ring_bytes: u64) -> Result<Layout, String> {
    if let Some(explicit) = &args.tier0_classes {
        let classes = parse_classes(explicit)?;
        if args.print_tier0_plan {
            let plan = capture_sizing::plan("", &classes, false, 0, ring_bytes)?;
            print_plan("explicit --tier0-classes", &plan, ring_bytes, 0);
        }
        return Ok((Some(classes), ring_bytes)); // Overrides deliberately bypass manifest sizing and its budget.
    }
    let baseline =
        continuum_sched::Profile::named(profile).ok_or_else(|| format!("no machine profile named {profile}"))?;
    let budget = args.tier0_camera_mib.checked_mul(1 << 20).ok_or("--tier0-camera-mib overflows bytes")?;
    let (source, text) = match &args.manifest {
        None => ("profile (no --manifest)".to_owned(), String::new()),
        Some(path) => match std::fs::read_to_string(path) {
            Ok(text) => (path.display().to_string(), text),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                warn!(manifest = %path.display(), "capture manifest is missing; sizing Tier 0 for no devices until device setup and broker restart");
                ("profile (missing manifest)".to_owned(), String::new())
            }
            Err(error) => return Err(format!("{}: {error}", path.display())),
        },
    };
    let room = if args.profile.is_some() { None } else { continuum_broker::tier0::shm_capacity() };
    let fit = capture_sizing::fit(&text, &baseline.slot_classes, cfg!(target_os = "macos"), budget, ring_bytes, room)
        .map_err(|error| if args.manifest.is_some() { format!("{source}: {error}") } else { error })?;
    let ring = if args.tier0_ring_mib.is_some() { ring_bytes } else { fit.ring_bytes };
    if fit.shrink > 0 {
        info!(
            shrink = fit.shrink,
            total_arena_bytes = fit.plan.total_bytes,
            shm_bytes = room.unwrap_or(0),
            "the profile's Tier 0 headroom doesn't fit half of /dev/shm; halved it this many times"
        );
    }
    if !fit.plan.cameras.is_empty() {
        info!(manifest = %source, cameras = fit.plan.cameras.len(), camera_bytes = fit.plan.camera_bytes,
            total_arena_bytes = fit.plan.total_bytes, ring_bytes = ring, "sized Tier 0 from capture settings");
    }
    if args.print_tier0_plan {
        print_plan(&source, &fit.plan, ring, fit.shrink);
    }
    Ok((Some(fit.plan.classes), ring))
}

fn print_plan(source: &str, plan: &capture_sizing::Plan, ring_bytes: u64, shrink: u32) {
    println!("tier0 source={source}");
    println!("profile_headroom_halvings={shrink}");
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
    continuum_log::init("continuum-broker");
    let args = Args::parse();
    let profile = args.profile.clone().unwrap_or_else(|| continuum_broker::server::default_profile().to_owned());
    let ring_mib = args.tier0_ring_mib.unwrap_or(continuum_broker::server::DEFAULT_RING_BYTES >> 20);
    let Some(ring_bytes) = ring_mib.checked_mul(1 << 20) else {
        error!("--tier0-ring-mib overflows bytes");
        return ExitCode::FAILURE;
    };
    let (tier0, ring_bytes) = match capture_layout(&args, &profile, ring_bytes) {
        Ok(layout) => layout,
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
            capture_layout(&args, "not-a-profile", 0).unwrap(),
            (Some(vec![continuum_sched::SlotClass { size_bytes: 16384, count: 4 }]), 0)
        );
    }

    #[test]
    fn invalid_explicit_classes_are_not_replaced_with_defaults() {
        let args = Args::parse_from(["broker", "--tier0-classes", "broken"]);
        assert!(capture_layout(&args, "mac_mini_64gb", 0).is_err());
    }

    #[test]
    fn a_named_profile_and_an_explicit_ring_region_are_used_whole() {
        // With --profile the headroom is not fitted to /dev/shm: the classes are the profile's, whatever this
        // machine's shared memory holds, and an explicit ring region is the one asked for.
        let args = Args::parse_from(["broker", "--profile", "linux_dual_3090", "--tier0-ring-mib", "3"]);
        let (classes, ring) = capture_layout(&args, "linux_dual_3090", 3 << 20).unwrap();
        assert_eq!(classes, Some(continuum_sched::Profile::linux_dual_3090().slot_classes));
        assert_eq!(ring, 3 << 20);
    }

    #[test]
    fn a_missing_manifest_is_sized_as_no_devices_and_an_unreadable_one_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("devices.toml");
        let args = Args::parse_from([
            std::ffi::OsString::from("broker"),
            "--profile".into(),
            "linux_dual_3090".into(),
            "--manifest".into(),
            missing.clone().into_os_string(),
        ]);
        let (classes, _) = capture_layout(&args, "linux_dual_3090", 0).unwrap();
        assert_eq!(classes, Some(continuum_sched::Profile::linux_dual_3090().slot_classes));
        std::fs::write(&missing, "bad [").unwrap();
        assert!(capture_layout(&args, "linux_dual_3090", 0).unwrap_err().contains("devices.toml"));
    }
}
