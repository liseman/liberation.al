//! Writes PCM-sized blocks into a ring of an existing arena, the way an audio capture worker will, for the live
//! ring tests in crates/continuum-py/tests.
//!
//! ```sh
//! cargo run --example pcm_producer -- <arena name> <ring> <blocks> <period_us>
//! ```
//!
//! Each record starts with the monotonic time it was published (CLOCK_MONOTONIC nanoseconds, u64 little-endian),
//! then its block number (u64), then zeros to the ring's stride. Blocks go out every `period_us` microseconds (0 for
//! as fast as possible). The writer never waits for a reader.

use std::process::ExitCode;
use std::thread;
use std::time::{Duration, Instant};

use continuum_shm::{Arena, Error, sys};

fn run(name: &str, ring: usize, blocks: u64, period: Duration) -> Result<(), Error> {
    let arena = Arena::open(name)?;
    let mut writer = arena.ring_writer(ring)?;
    if writer.stride() < 16 {
        return Err(Error::BadConfig(format!("ring {ring}'s stride is {}; the stamp needs 16 bytes", writer.stride())));
    }
    let mut record = vec![0u8; writer.stride()];
    let start = Instant::now();
    for n in 0..blocks {
        record[8..16].copy_from_slice(&n.to_le_bytes());
        record[..8].copy_from_slice(&sys::monotonic_ns().to_le_bytes()); // stamped last, right before publishing
        writer.push(&record)?;
        if !period.is_zero() {
            let due = start + period * u32::try_from(n + 1).unwrap_or(u32::MAX);
            thread::sleep(due.saturating_duration_since(Instant::now()));
        }
    }
    eprintln!("pcm_producer: {blocks} blocks in {:.3} s", start.elapsed().as_secs_f64());
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let [_, name, ring, blocks, period_us] = args.as_slice() else {
        eprintln!("usage: pcm_producer <arena name> <ring> <blocks> <period_us>");
        return ExitCode::from(2);
    };
    let (Ok(ring), Ok(blocks), Ok(period_us)) = (ring.parse(), blocks.parse(), period_us.parse()) else {
        eprintln!("ring, blocks, and period_us must be whole numbers");
        return ExitCode::from(2);
    };
    match run(name, ring, blocks, Duration::from_micros(period_us)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("pcm_producer: {e}");
            ExitCode::FAILURE
        }
    }
}
