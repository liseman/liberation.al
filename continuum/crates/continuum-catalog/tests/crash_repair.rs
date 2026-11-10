//! redb's crash repair at a realistic size: a writer is killed with SIGKILL mid-commit on a
//! catalog of a few GB, and the reopen, which runs redb's repair, must succeed with every acknowledged commit in it.
//! Ignored by default (it writes GBs and takes minutes); run it by hand, as docs/CONTINUUM.md says:
//!
//! ```text
//! CONTINUUM_CRASH_DIR=/some/disk/dir tools/build-slot.sh nice -n 19 ionice -c3 \
//!   cargo test --release --manifest-path continuum/Cargo.toml -p continuum-catalog --test crash_repair -- \
//!   --ignored --nocapture
//! ```
//!
//! `CONTINUUM_CRASH_GB` (3) sizes the catalog, built once and reused; `CONTINUUM_CRASH_ROUNDS` (5) sets the
//! rounds. A failed reopen stops the test and leaves the file in place for inspection.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use continuum_catalog::*;
use continuum_core::{HeaderBuf, HeaderSpec, StorageLocator, StreamId, Uuid};

const TEST: &str = "redb_repairs_a_large_catalog_after_kill_9";
const CHILD: &str = "CONTINUUM_CRASH_CHILD";
const T0: u64 = 1_700_000_000_000_000_000;

fn id(n: u64) -> Uuid {
    let mut b = [0u8; 16];
    b[0] = 0x01;
    b[6] = 0x70;
    b[8..16].copy_from_slice(&(n | 1 << 63).to_be_bytes());
    Uuid(b)
}

fn stream(c: &RedbCatalog) -> StreamId {
    c.register_stream(StreamInfo {
        name: "crash.repair".into(),
        kind: StreamKind::Sensor,
        layer: continuum_core::DomainLayer::Layer1Raw,
        format: continuum_core::PayloadFormat::RawBytes,
        shape_name: "crash".into(),
        shape_version: 1,
        rate_hz: Some(100.0),
        anchor: None,
    })
    .unwrap()
}

fn payload(n: u64, s: StreamId, bytes: usize) -> NewPayload {
    let start = T0 + n * 10_000_000;
    let spec = HeaderSpec::raw(id(n), s, start, start + 1);
    NewPayload {
        header: HeaderBuf::build(&spec).unwrap(),
        location: Some(StorageLocator::Inline { bytes: vec![b'x'; bytes] }),
        coverage: None,
    }
}

fn catalog(dir: &Path) -> PathBuf {
    dir.join("catalog.redb")
}

/// Payloads 0 to the returned count fill the catalog to `gb`, measured after a clean close (redb grows the file a
/// region at a time and trims the free tail when it closes). Built once.
fn build(dir: &Path, gb: u64) -> u64 {
    let marker = dir.join("built");
    if let Some(n) = std::fs::read_to_string(&marker).ok().and_then(|s| s.trim().parse().ok()) {
        return n;
    }
    let mut n = 0u64;
    let began = Instant::now();
    while std::fs::metadata(catalog(dir)).map_or(0, |m| m.len()) < gb << 30 {
        let c = RedbCatalog::open(catalog(dir)).unwrap();
        let s = stream(&c);
        // About a tenth of the target at a time, then a clean close to see what it really takes.
        let goal = n + (gb << 30) / 10 / 1600;
        while n < goal {
            c.insert((n..n + 2048).map(|k| payload(k, s, 1500)).collect()).unwrap();
            n += 2048;
        }
        c.close(Duration::from_secs(30)).unwrap();
    }
    std::fs::write(&marker, n.to_string()).unwrap();
    eprintln!("built {} GB, {n} payloads, in {:.0} s", gb, began.elapsed().as_secs_f64());
    n
}

/// The child: commits batches of 8 from `from` until it is killed, saying how far it got after each commit.
fn child(dir: &Path, from: u64) -> ! {
    let c = RedbCatalog::open(catalog(dir)).unwrap();
    let s = stream(&c);
    let mut n = from;
    loop {
        c.insert((n..n + 8).map(|k| payload(k, s, 300)).collect()).unwrap();
        n += 8;
        let tmp = dir.join("progress.tmp");
        std::fs::write(&tmp, n.to_string()).unwrap();
        std::fs::rename(&tmp, dir.join("progress")).unwrap();
    }
}

fn progress(dir: &Path) -> Option<u64> {
    std::fs::read_to_string(dir.join("progress")).ok().and_then(|s| s.trim().parse().ok())
}

#[test]
#[ignore = "writes GBs and takes minutes: run by hand with CONTINUUM_CRASH_DIR (docs/CONTINUUM.md)"]
fn redb_repairs_a_large_catalog_after_kill_9() {
    let dir = PathBuf::from(std::env::var("CONTINUUM_CRASH_DIR").expect("CONTINUUM_CRASH_DIR: a scratch directory"));
    if let Ok(from) = std::env::var(CHILD) {
        child(&dir, from.parse().unwrap());
    }
    assert!(!dir.to_string_lossy().contains("data/state"), "never the live data directory");
    std::fs::create_dir_all(&dir).unwrap();
    let var = |name: &str, default: u64| std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
    let rounds = var("CONTINUUM_CRASH_ROUNDS", 5);
    let mut from = build(&dir, var("CONTINUUM_CRASH_GB", 3)).max(progress(&dir).unwrap_or(0));
    let me = std::env::current_exe().unwrap();
    for round in 1..=rounds {
        let _ = std::fs::remove_file(dir.join("progress"));
        let mut writer = Command::new(&me)
            .args([TEST, "--exact", "--ignored", "--nocapture", "--test-threads=1"])
            .env(CHILD, from.to_string())
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        // Wait for its first commits, then kill it at an uneven point mid-stream.
        let waited = Instant::now();
        while progress(&dir).is_none_or(|p| p < from + 64) {
            assert!(waited.elapsed() < Duration::from_secs(600), "round {round}: the writer never committed");
            assert!(writer.try_wait().unwrap().is_none(), "round {round}: the writer exited");
            std::thread::sleep(Duration::from_millis(20));
        }
        std::thread::sleep(Duration::from_millis(150 + round * 113 % 400));
        writer.kill().unwrap(); // SIGKILL
        writer.wait().unwrap();
        let acked = progress(&dir).unwrap();
        let (len, began) = (std::fs::metadata(catalog(&dir)).unwrap().len(), Instant::now());
        let opened = RedbCatalog::open(catalog(&dir));
        let repair = began.elapsed();
        let c = match opened {
            Ok(c) => c,
            Err(e) => panic!(
                "round {round}: reopen failed after {:.1} s: {e}; the file is kept at {}",
                repair.as_secs_f64(),
                catalog(&dir).display()
            ),
        };
        // Every acknowledged batch is there, and what came after is whole batches or nothing.
        for k in [0, from, acked - 8, acked - 1] {
            assert!(c.get(id(k)).unwrap().is_some(), "round {round}: acknowledged payload {k} lost");
        }
        let mut beyond = acked;
        while c.get(id(beyond)).unwrap().is_some() {
            beyond += 1;
        }
        assert!((beyond - acked).is_multiple_of(8), "round {round}: a torn batch after {acked}");
        eprintln!(
            "round {round}: file {:.2} GB, repair and open {:.1} s, acknowledged {acked}, landed {beyond}: ok",
            f64::from(u32::try_from(len >> 20).unwrap_or(u32::MAX)) / 1024.0,
            repair.as_secs_f64(),
        );
        // A clean close, so the next writer starts without a repair of its own.
        c.close(Duration::from_secs(30)).unwrap();
        drop(c);
        from = beyond;
        std::fs::write(dir.join("progress"), from.to_string()).unwrap();
    }
}
