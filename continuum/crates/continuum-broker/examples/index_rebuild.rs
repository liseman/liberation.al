//! Bounded availability-index startup measurement using only a fresh synthetic redb catalog.
//!
//! Run from `continuum/` through the shared Rust build slot, at standard priority:
//! `../tools/build-slot.sh cargo run --release -p continuum-broker --example index_rebuild -- --guard`
//! The default measures 1M then 2M gapped payloads on one stream, the worst case for prefix rescanning.
//! Catalog creation is timed separately. The temporary catalog is deleted when the example exits.

//! For individual phases, explicitly select the synthetic-only mode and scratch parent:
//! `../tools/build-slot.sh cargo run --release -p continuum-broker --example index_rebuild -- --startup-phases --scratch /path/to/scratch --payloads 100000 --streams 100`
//! This mode creates a new temporary subtree, excludes seed/clean-close from startup timings, and emits JSON.
//! It accepts no existing catalog and cannot be combined with --guard.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::Parser;
use continuum_broker::{index, tier0::Tier0};
use continuum_catalog::{Catalog, CoverageMark, NewPayload, RedbCatalog, StreamInfo, StreamKind};
use continuum_core::{
    DomainLayer, HeaderBuf, HeaderSpec, Interval, PAGE_ALIGN, PayloadFormat, StageId, StorageLocator, StorageTier, Uuid,
};
use continuum_sched::SlotClass;
use serde_json::json;

#[derive(Parser)]
struct Args {
    /// Rows in phase mode, or the larger rebuild size (the legacy first pass uses half).
    #[arg(long, default_value_t = 2_000_000)]
    payloads: u32,
    /// Catalog page size, matching broker startup by default.
    #[arg(long, default_value_t = 1024)]
    page: usize,
    /// Require the larger rebuild under 30s and at most 2.6 times the smaller one.
    #[arg(long)]
    guard: bool,
    /// Measure individual startup phases on a fresh synthetic catalog instead of the legacy doubling guard.
    #[arg(long, conflicts_with = "guard")]
    startup_phases: bool,
    /// Also measure valid checkpoint restoration and stale-checkpoint fallback, on the same fresh fixture.
    #[arg(long, requires = "startup_phases")]
    restart_checkpoint: bool,
    /// Existing scratch directory, required for --startup-phases.
    #[arg(long, required_if_eq("startup_phases", "true"))]
    scratch: Option<PathBuf>,
    /// Registered streams in --startup-phases mode.
    #[arg(long, default_value_t = 100)]
    streams: u32,
    /// Inline bytes per generated payload in --startup-phases mode.
    #[arg(long, default_value_t = 128)]
    inline_bytes: u32,
    /// Payloads per seed transaction in --startup-phases mode.
    #[arg(long, default_value_t = 4096)]
    batch_size: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.startup_phases {
        return profile_startup(&args);
    }
    if args.payloads < 2 || args.page == 0 {
        return Err("payloads must be at least two and page must be positive".into());
    }
    let temp = tempfile::tempdir()?;
    let catalog = RedbCatalog::open(temp.path().join("synthetic.redb"))?;
    let stream = catalog.register_stream(StreamInfo {
        name: "synthetic.video".into(),
        kind: StreamKind::Sensor,
        layer: DomainLayer::Layer1Raw,
        format: PayloadFormat::RawBytes,
        shape_name: "synthetic".into(),
        shape_version: 1,
        rate_hz: Some(30.0),
        anchor: None,
    })?;
    let mut previous = 0;
    let mut times = Vec::new();
    for count in [args.payloads / 2, args.payloads] {
        let preparing = Instant::now();
        for start in (previous..count).step_by(10_000) {
            let mut batch = Vec::new();
            for n in start..count.min(start.saturating_add(10_000)) {
                let mut id = [0; 16];
                id[12..].copy_from_slice(&n.to_be_bytes());
                let at = u64::from(n) * 100_000_000;
                let mut spec = HeaderSpec::raw(Uuid(id), stream, at, at);
                spec.seq = n;
                batch.push(NewPayload {
                    header: HeaderBuf::build(&spec)?,
                    location: Some(StorageLocator::Chunk {
                        tier: StorageTier::Tier1Warm,
                        uri: "synthetic-only".into(),
                        offset: u64::from(n),
                        length: 1,
                    }),
                    coverage: None,
                });
            }
            catalog.insert(batch)?;
        }
        let prepare = preparing.elapsed();
        let started = Instant::now();
        let (index, stats) = index::load(&catalog, args.page)?;
        let elapsed = started.elapsed();
        assert_eq!(stats.payloads, count as usize);
        assert_eq!(stats.segments, count as usize);
        assert!(stats.stale_tier0.is_empty());
        println!(
            "payloads={count} segments={} page={} prepare_s={:.3} rebuild_s={:.3}",
            stats.segments,
            args.page,
            prepare.as_secs_f64(),
            elapsed.as_secs_f64()
        );
        drop(index);
        times.push(elapsed);
        previous = count;
    }
    let ratio = times[1].as_secs_f64() / times[0].as_secs_f64();
    println!("doubling_ratio={ratio:.3}");
    if args.guard && (times[1] >= Duration::from_secs(30) || ratio > 2.6) {
        return Err("startup rebuild exceeded the 30s or 2.6x doubling budget".into());
    }
    Ok(())
}

fn id(n: u32) -> Uuid {
    let mut bytes = [0; 16];
    bytes[6] = 0x70;
    bytes[8] = 0x80;
    bytes[12..].copy_from_slice(&n.to_be_bytes());
    Uuid(bytes)
}

fn seconds(t: Duration) -> f64 {
    t.as_secs_f64()
}

fn profile_startup(args: &Args) -> Result<(), Box<dyn Error>> {
    if !(1..=10_000_000).contains(&args.payloads)
        || !(1..=1000).contains(&args.streams)
        || args.inline_bytes > 65_536
        || !(1..=65_536).contains(&args.batch_size)
        || args.page == 0
        || args.guard
    {
        return Err("startup phases require payloads 1..=10000000, streams 1..=1000, inline bytes <=65536, batch size 1..=65536, positive page, and no --guard".into());
    }
    let scratch = args.scratch.as_ref().ok_or("--startup-phases requires --scratch")?;
    // A process-local stderr subscriber; never install the service's global logger or select a log path.
    let subscriber = tracing_subscriber::fmt().with_writer(std::io::stderr).with_ansi(false).finish();
    let _logging = tracing::subscriber::set_default(subscriber);
    if !scratch.is_dir() {
        return Err("--scratch must name an existing directory".into());
    }
    let temporary = tempfile::Builder::new().prefix("continuum-startup-synthetic-").tempdir_in(scratch)?;
    let path = temporary.path().join("synthetic.redb");
    let seed_start = Instant::now();
    let seed = RedbCatalog::open(&path)?;
    let mut streams = Vec::new();
    for n in 0..args.streams {
        streams.push(seed.register_stream(StreamInfo {
            name: format!("synthetic.{n}"),
            kind: StreamKind::Derived,
            layer: DomainLayer::Layer1Raw,
            format: PayloadFormat::RawBytes,
            shape_name: String::new(),
            shape_version: 0,
            rate_hz: None,
            anchor: None,
        })?);
    }
    let (mut header_bytes, mut inline_bytes, mut header_only, mut stale) = (0u64, 0u64, 0u64, 0u64);
    let mut batch = Vec::with_capacity(args.batch_size as usize);
    for n in 0..args.payloads {
        let stream = streams[(n % args.streams) as usize];
        // Sparse instants, globally sorted by time and separated by empty nanoseconds.
        let time = (u64::from(n) + 1) * 1_000_000;
        let header = HeaderBuf::build(&HeaderSpec::raw(id(n + 1), stream, time, time))?;
        header_bytes += header.as_bytes().len() as u64;
        let location = if n % 512 == 0 {
            stale += 1;
            Some(StorageLocator::Shm {
                arena: "synthetic-stale-never-opened".into(),
                arena_generation: 1,
                slab_class: 0,
                slot: n,
                slot_generation: 1,
                offset: 0,
                length: 1,
            })
        } else if n % 8 == 0 {
            header_only += 1;
            None
        } else {
            inline_bytes += u64::from(args.inline_bytes);
            Some(StorageLocator::Inline { bytes: vec![0x5a; args.inline_bytes as usize] })
        };
        batch.push(NewPayload { header, location, coverage: None });
        if batch.len() == args.batch_size as usize {
            seed.insert(std::mem::replace(&mut batch, Vec::with_capacity(args.batch_size as usize)))?;
        }
    }
    if !batch.is_empty() {
        seed.insert(batch)?;
    }
    seed.close(Duration::from_secs(30))?;
    drop(seed);
    let seed_seconds = seconds(seed_start.elapsed());
    let file_bytes = std::fs::metadata(&path)?.len();

    let catalog = RedbCatalog::open(&path)?;
    let open = catalog.open_timings();
    let (index, stats, load) = index::load_profiled(&catalog, args.page)?;
    assert_eq!(stats.streams, usize::try_from(args.streams)?);
    assert_eq!(stats.payloads, usize::try_from(args.payloads)?);
    assert_eq!(u64::try_from(stats.segments)?, u64::from(args.payloads) - header_only - stale);
    assert_eq!(u64::try_from(stats.stale_tier0.len())?, stale);
    assert_eq!(stats.coverage_intervals, 0);

    let start = Instant::now();
    let cleared = index::clear_stale_tier0(&catalog, &stats.stale_tier0)?;
    let clear_seconds = seconds(start.elapsed());
    assert_eq!(u64::try_from(cleared)?, stale);
    let start = Instant::now();
    let pending = catalog.pending_purges()?.len();
    let pending_seconds = seconds(start.elapsed());
    let start = Instant::now();
    let keepers = catalog.keepers()?.len();
    let keepers_seconds = seconds(start.elapsed());
    let start = Instant::now();
    // The unique temporary socket path isolates arena names and generation files.
    let arena = Tier0::create(
        &temporary.path().join("synthetic.sock"),
        &[SlotClass { size_bytes: u64::try_from(PAGE_ALIGN)?, count: 8 }],
        64 * 1024,
    )?;
    let arena_seconds = seconds(start.elapsed());
    let arena_bytes = arena.summary().size_bytes;
    drop(arena); // Tier0::drop unlinks its unique shared memory name.
    let checkpoint = if args.restart_checkpoint {
        Some(profile_checkpoint(catalog, index, &path, args.page, &stats.stale_tier0)?)
    } else {
        drop(index);
        catalog.close(Duration::from_secs(30))?;
        drop(catalog);
        None
    };
    let output = json!({
        "synthetic_only": true,
        "restart_checkpoint": checkpoint,
        "config": {"payloads": args.payloads, "streams": args.streams, "inline_bytes": args.inline_bytes,
            "batch_size": args.batch_size, "index_page": args.page},
        "counts": {"streams": stats.streams, "payloads": stats.payloads, "segments": stats.segments,
            "coverage_intervals": stats.coverage_intervals, "header_only": header_only,
            "stale_tier0_seeded": stale, "stale_tier0_cleared": cleared, "pending_purges": pending, "keepers": keepers},
        "bytes": {"catalog_after_seed": file_bytes, "serialized_headers": header_bytes,
            "inline_payloads": inline_bytes, "arena": arena_bytes},
        "seconds": {"seed_and_clean_close": seed_seconds,
            "open": {"scrub_recovery": seconds(open.scrub_recovery), "database_open": seconds(open.database_open),
                "schema_setup": seconds(open.schema_setup), "recording_repair": seconds(open.recording_repair), "total": seconds(open.total)},
            "index": {"stream_reads": seconds(load.stream_reads), "payload_reads": seconds(load.payload_reads),
                "payload_apply": seconds(load.payload_apply), "coverage_load": seconds(load.coverage_load), "total": seconds(load.total)},
            "clear_stale_tier0": clear_seconds, "pending_purges": pending_seconds,
            "keepers": keepers_seconds, "arena_create": arena_seconds}
    });
    temporary.close()?;
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

fn profile_checkpoint(
    catalog: RedbCatalog,
    index: index::AvailabilityIndex,
    path: &Path,
    page: usize,
    hot_ids: &[Uuid],
) -> Result<serde_json::Value, Box<dyn Error>> {
    let start = Instant::now();
    let _ = catalog.restart_stamp()?.ok_or("synthetic catalog has no restart stamp")?;
    let initialize = seconds(start.elapsed());
    // Model a clean live run that still has obsolete hot copies beside its durable projection.
    // This fixture preparation is outside both checkpoint and restore clocks; it adds no payload rows.
    let hot_setup_start = Instant::now();
    for (slot, &id) in hot_ids.iter().enumerate() {
        catalog.set_location(
            id,
            StorageLocator::Shm {
                arena: "synthetic-checkpoint-hot-never-opened".into(),
                arena_generation: 1,
                slab_class: 0,
                slot: u32::try_from(slot)?,
                slot_generation: 1,
                offset: 0,
                length: 1,
            },
        )?;
    }
    let hot_setup = seconds(hot_setup_start.elapsed());
    let stamp = catalog.restart_stamp()?.ok_or("synthetic catalog lost its restart stamp")?;
    let checkpoint_started = Instant::now();
    let expected_size = index.size();
    let start = Instant::now();
    let encoded = index.restart_projection_bytes()?;
    let encode = seconds(start.elapsed());
    let bytes = encoded.len();
    let start = Instant::now();
    assert!(catalog.save_restart_projection(stamp, &encoded)?);
    let publish = seconds(start.elapsed());
    drop(index);
    let start = Instant::now();
    catalog.close(Duration::from_secs(30))?;
    drop(catalog);
    let close = seconds(start.elapsed());
    let checkpoint_total = seconds(checkpoint_started.elapsed());

    let start = Instant::now();
    let catalog = RedbCatalog::open(path)?;
    let reopen = seconds(start.elapsed());
    let start = Instant::now();
    let cached = catalog
        .load_restart_projection(continuum_catalog::MAX_RESTART_PROJECTION_BYTES)?
        .map_err(|miss| format!("clean synthetic checkpoint was rejected after reopen: {miss}"))?;
    let read = seconds(start.elapsed());
    assert_eq!(cached.stamp, stamp);
    let mut expected_hot = hot_ids.to_vec();
    expected_hot.sort();
    assert_eq!(cached.hot_ids, expected_hot);
    let start = Instant::now();
    let restored = index::AvailabilityIndex::restore_restart_projection(&cached.bytes)?;
    let decode = seconds(start.elapsed());
    let start = Instant::now();
    let cleared = index::clear_stale_tier0(&catalog, &cached.hot_ids)?;
    assert_eq!(cleared, hot_ids.len());
    assert!(catalog.pending_purges()?.is_empty());
    assert!(catalog.keepers()?.is_empty());
    let auxiliary = seconds(start.elapsed());
    assert_eq!(restored.size(), expected_size);
    // Canonical parity validation is excluded from the measured restore path.
    assert_eq!(restored.restart_projection_bytes()?, encoded);
    // Hot cleanup intentionally invalidates the old stamp. Re-seal only for this independent
    // coverage-only invalidation measurement, outside the measured startup path.
    let reseal_start = Instant::now();
    let current = catalog.restart_stamp()?.ok_or("synthetic catalog lost its restart stamp")?;
    assert!(catalog.save_restart_projection(current, &cached.bytes)?);
    let reseal = seconds(reseal_start.elapsed());
    drop(encoded);
    drop(cached);
    drop(restored);

    let stream = catalog.streams(None, 1)?.first().ok_or("synthetic fixture has no stream")?.0;
    let invalidate_start = Instant::now();
    catalog.add_coverage(CoverageMark { source: stream, stage: StageId(1), span: Interval::new(0, 1).unwrap() })?;
    let invalidate = seconds(invalidate_start.elapsed());
    assert!(
        catalog.load_restart_projection(continuum_catalog::MAX_RESTART_PROJECTION_BYTES)?.is_err(),
        "a coverage-only write must invalidate without changing payload sequence"
    );
    let (fallback, fallback_stats, fallback_time) = index::load_profiled(&catalog, page)?;
    assert_eq!(fallback_stats.segments, expected_size.0);
    assert_eq!(fallback_stats.coverage_intervals, expected_size.1 + 1);
    assert!(fallback_stats.stale_tier0.is_empty());
    drop(fallback);
    catalog.close(Duration::from_secs(30))?;
    drop(catalog);
    Ok(json!({
        "projection_bytes": bytes,
        "restored_hot_ids_cleared": cleared,
        "restore_path": "checkpoint_table_and_hot_id_index",
        "exact_canonical_roundtrip": true,
        "coverage_change_invalidates": true,
        "seconds": {"initialize_tracking_once": initialize, "prepare_hot_copies_outside_timing": hot_setup, "encode": encode,
            "publish": publish, "clean_close": close,
            "checkpoint_and_close": checkpoint_total, "checkpoint_component_sum": encode + publish + close,
            "validation_reseal_outside_timing": reseal, "invalidate": invalidate,
            "reopen": reopen, "read": read, "decode": decode, "auxiliary": auxiliary,
            "valid_restore_total": reopen + read + decode + auxiliary,
            "fallback_index": fallback_time.total.as_secs_f64()}
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startup_phases_require_an_explicit_scratch_parent_and_refuse_the_legacy_guard() {
        assert!(Args::try_parse_from(["index_rebuild", "--startup-phases"]).is_err());
        assert!(Args::try_parse_from(["index_rebuild", "--startup-phases", "--scratch", ".", "--guard"]).is_err());
    }

    #[test]
    fn checkpoint_mode_requires_the_explicit_synthetic_phase_mode() {
        assert!(Args::try_parse_from(["index_rebuild", "--restart-checkpoint"]).is_err());
        assert!(
            Args::try_parse_from(["index_rebuild", "--restart-checkpoint", "--startup-phases", "--scratch", "."])
                .is_ok()
        );
    }

    #[test]
    fn legacy_guard_keeps_its_original_defaults_without_scratch() {
        let args = Args::try_parse_from(["index_rebuild", "--guard"]).unwrap();
        assert_eq!(args.payloads, 2_000_000);
        assert_eq!(args.page, 1024);
        assert!(args.guard);
        assert!(!args.startup_phases);
        assert!(args.scratch.is_none());
        assert!(!args.restart_checkpoint);
    }
}
