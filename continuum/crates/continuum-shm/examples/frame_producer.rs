//! Writes 4K NV12 frames into an existing arena, the way a capture worker will, for the cross-process test.
//!
//! ```sh
//! cargo run --example frame_producer -- <arena name> <frames> <fps>
//! ```
//!
//! The arena must have slab class 0 sized for a 4K NV12 frame and ring 0 with a 32-byte stride (the Python test in
//! crates/continuum-py/tests creates it). Each frame claims a slot with `claim_overwriting`, fills it with the frame
//! number's low byte (the first 8 bytes hold the whole frame number), publishes it, and pushes a record to ring 0:
//! the slot handle (12 bytes), 4 zero bytes, the frame number (u64), and the length (u64), all little-endian. A last
//! record with frame number `u64::MAX` marks the end. Frames are paced at `fps`, so the run is short and bounded.

use std::process::ExitCode;
use std::thread;
use std::time::{Duration, Instant};

use continuum_shm::{Arena, Error, SlotHandle};

const WIDTH: usize = 3840;
const HEIGHT: usize = 2160;
const NV12_LEN: usize = WIDTH * HEIGHT * 3 / 2;
const STREAM_ID: u64 = 1;
const END: u64 = u64::MAX;

fn record(handle: SlotHandle, frame: u64, len: u64) -> [u8; 32] {
    let mut r = [0u8; 32];
    r[..SlotHandle::ENCODED_LEN].copy_from_slice(&handle.to_le_bytes());
    r[16..24].copy_from_slice(&frame.to_le_bytes());
    r[24..32].copy_from_slice(&len.to_le_bytes());
    r
}

fn run(name: &str, frames: u64, fps: u32) -> Result<(), Error> {
    let arena = Arena::open(name)?;
    let mut ring = arena.ring_writer(0)?;
    let period = Duration::from_secs(1) / fps;
    let start = Instant::now();
    let mut gaps = 0;
    for frame in 0..frames {
        let (mut slot, overwritten) = arena.claim_overwriting(0, STREAM_ID)?;
        gaps += overwritten.len();
        let bytes = &mut slot.as_mut_slice()[..NV12_LEN];
        bytes.fill(frame as u8);
        bytes[..8].copy_from_slice(&frame.to_le_bytes());
        let handle = slot.publish(NV12_LEN)?;
        ring.push(&record(handle, frame, NV12_LEN as u64))?;
        let due = start + period * u32::try_from(frame + 1).unwrap_or(u32::MAX);
        thread::sleep(due.saturating_duration_since(Instant::now()));
    }
    ring.push(&record(SlotHandle { class: 0, index: 0, generation: 0 }, END, 0))?;
    eprintln!(
        "frame_producer: {frames} frames in {:.2} s, {gaps} overwritten unflushed (expected: no tier worker)",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let [_, name, frames, fps] = args.as_slice() else {
        eprintln!("usage: frame_producer <arena name> <frames> <fps>");
        return ExitCode::from(2);
    };
    let (Ok(frames), Ok(fps)) = (frames.parse(), fps.parse()) else {
        eprintln!("frames and fps must be whole numbers");
        return ExitCode::from(2);
    };
    match run(name, frames, fps) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("frame_producer: {e}");
            ExitCode::FAILURE
        }
    }
}
