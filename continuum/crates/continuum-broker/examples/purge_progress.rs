//! Measure one routine trim on temporary synthetic state, without live traffic.
//!
//! From `continuum/`, use the shared standard-priority slot:
//! `../tools/build-slot.sh cargo run --release -p continuum-broker --example purge_progress`
//! The default is 15,000 old live records with retained parents and a 60s tail. It applies the writer's ordinary
//! cooperative purge work window, separates fixture creation, and checks removal boundaries. No broker, device, or keeper is used.

use std::collections::BTreeMap;
use std::time::Instant;

use clap::Parser;
use continuum_broker::core::WriterConfig;
use continuum_catalog::{Catalog, NewPayload, PurgeProgress, PurgeSelector, RedbCatalog, StreamInfo, StreamKind};
use continuum_core::{DomainLayer, HeaderBuf, HeaderSpec, Interval, PayloadFormat, StorageLocator, StreamId, Uuid};

#[derive(Parser)]
struct Args {
    /// How many old live records the trim removes.
    #[arg(long, default_value_t = 15_000)]
    records: u32,
}

fn id(n: u32, kind: u8) -> Uuid {
    let mut bytes = [0; 16];
    bytes[6] = 0x70;
    bytes[8] = 0x80;
    bytes[9] = kind;
    bytes[12..].copy_from_slice(&n.to_be_bytes());
    Uuid(bytes)
}

fn stream(c: &RedbCatalog, name: &str) -> Result<StreamId, Box<dyn std::error::Error>> {
    Ok(c.register_stream(StreamInfo {
        name: name.into(),
        kind: StreamKind::Derived,
        layer: DomainLayer::Layer1Raw,
        format: PayloadFormat::Text,
        shape_name: name.into(),
        shape_version: 1,
        rate_hz: None,
        anchor: None,
    })?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.records == 0 || args.records > 1_000_000 {
        return Err("records must be between 1 and 1000000".into());
    }
    let temp = tempfile::tempdir()?;
    let c = RedbCatalog::open(temp.path().join("synthetic.redb"))?;
    let parent = stream(&c, "synthetic.frames")?;
    let live = stream(&c, "synthetic.live")?;
    let tail = (args.records / 10).max(1);
    let until = 600_000_000_000;
    let preparing = Instant::now();
    for start in (0..args.records + tail).step_by(1000) {
        let mut batch = Vec::new();
        for n in start..(args.records + tail).min(start + 1000) {
            let at = u64::from(n) * until / u64::from(args.records);
            batch.push(NewPayload {
                header: HeaderBuf::build(&HeaderSpec::raw(id(n, 1), parent, at, at))?,
                location: None,
                coverage: None,
            });
            let mut spec = HeaderSpec::raw(id(n, 2), live, at, at);
            spec.parents = vec![id(n, 1)];
            batch.push(NewPayload {
                header: HeaderBuf::build(&spec)?,
                location: Some(StorageLocator::Inline { bytes: b"{}".to_vec() }),
                coverage: None,
            });
        }
        c.insert(batch)?;
    }
    let prepare = preparing.elapsed();
    let config = WriterConfig::default();
    let purge = id(0, 3);
    let started = Instant::now();
    c.start_purge(
        PurgeSelector::Range { span: Interval::new(0, until).unwrap(), streams: Some(vec![live]) },
        "synthetic trim",
        "",
        purge,
        0,
    )?;
    let mut steps = 0;
    let mut phases = BTreeMap::new();
    loop {
        let step = Instant::now();
        let result = c.purge_batch(purge, config.max_budget, &mut || step.elapsed() >= config.step_target)?;
        steps += 1;
        match result {
            PurgeProgress::Running { phase, .. } => *phases.entry(format!("{phase:?}")).or_insert(0) += 1,
            PurgeProgress::Removed { report, .. } => {
                assert_eq!(report.removed.len(), args.records as usize);
                break;
            }
        }
    }
    let elapsed = started.elapsed();
    assert!(c.get(id(0, 1))?.is_some(), "parents remain");
    assert!(c.get(id(0, 2))?.is_none(), "old live records leave");
    assert!(c.get(id(args.records, 2))?.is_some(), "the tail remains");
    println!(
        "records={} prepare_s={:.3} catalog_s={:.3} steps={steps} phases={phases:?}",
        args.records,
        prepare.as_secs_f64(),
        elapsed.as_secs_f64()
    );
    Ok(())
}
