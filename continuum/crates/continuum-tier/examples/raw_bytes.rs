//! How many bytes an hour a raw camera costs in Tier 1: synthetic 1280x720 UYVY at 30 fps (a slowly panning
//! scene with sensor-like noise, which is what defeats LZ4) through LZ4, and through each H.264 encoder named on the
//! command line with the raw camera settings (`continuum_tier::raw::RawEncoder`: 1 s GOP, low delay, limited range).
//!
//! `cargo run --release --features ffmpeg --example raw_bytes -- SECONDS [ENCODER ...]`
//! (encoders like `h264_nvenc` and `libx264`; none given measures LZ4 only). Short runs: it extrapolates to an hour.

use std::time::Instant;

use continuum_tier::codecs::lz4_compress;
use continuum_tier::raw::{RawEncoder, RawLayout};

const W: u32 = 1280;
const H: u32 = 720;
const FPS: u32 = 30;

/// Frame `i`: a gradient panning a pixel a frame, a bright square drifting across, and noise of a few levels on luma.
fn frame(i: u32, seed: &mut u64) -> Vec<u8> {
    let mut out = vec![0u8; (W * 2 * H) as usize];
    let (sx, sy) = ((i * 4) % (W - 120), 300 + (i % 60));
    for y in 0..H {
        for x in (0..W).step_by(2) {
            let base = |x: u32| -> i32 {
                let g = ((x + i) % 256) as i32 / 2 + (y as i32 * 80 / H as i32);
                if x >= sx && x < sx + 120 && y >= sy && y < sy + 120 { 220 } else { 40 + g }
            };
            let mut noise = || {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                (*seed % 7) as i32 - 3
            };
            let at = ((y * W * 2) + x * 2) as usize;
            out[at] = 128; // U
            out[at + 1] = (base(x) + noise()).clamp(16, 235) as u8; // Y0
            out[at + 2] = 128; // V
            out[at + 3] = (base(x + 1) + noise()).clamp(16, 235) as u8; // Y1
        }
    }
    out
}

fn per_hour(bytes: u64, frames: u32) -> f64 {
    bytes as f64 / f64::from(frames) * f64::from(FPS) * 3600.0
}

fn main() {
    let mut args = std::env::args().skip(1);
    let seconds: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(10);
    let encoders: Vec<String> = args.collect();
    let frames = seconds * FPS;
    let layout = RawLayout::of("UYVY", W, H, 0).expect("a layout");
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let raw: Vec<Vec<u8>> = (0..frames).map(|i| frame(i, &mut seed)).collect();
    let raw_bytes: u64 = raw.iter().map(|f| f.len() as u64).sum();
    let lz4: u64 = raw.iter().map(|f| lz4_compress(f).len() as u64).sum();
    println!("{seconds} s of {W}x{H} UYVY at {FPS} fps");
    println!("raw:  {:>8.1} GB/hour", per_hour(raw_bytes, frames) / 1e9);
    println!("lz4:  {:>8.1} GB/hour", per_hour(lz4, frames) / 1e9);
    for name in &encoders {
        let mut enc = match RawEncoder::open(layout, FPS, &[name.as_str()]) {
            Ok(e) => e,
            Err(e) => {
                println!("{name}: doesn't open here: {e}");
                continue;
            }
        };
        let t = Instant::now();
        let mut bytes = 0u64;
        let mut units = 0u32;
        for (i, f) in raw.iter().enumerate() {
            let start = u64::try_from(i).unwrap() * 33_333_333;
            for au in enc.encode(f, start, start).expect("encode") {
                bytes += au.bytes.len() as u64;
                units += 1;
            }
        }
        for au in enc.finish().expect("finish") {
            bytes += au.bytes.len() as u64;
            units += 1;
        }
        let took = t.elapsed().as_secs_f64();
        println!(
            "{name}: {:>8.2} GB/hour ({:.1} Mbit/s), {units} units, {:.0} fps encoded",
            per_hour(bytes, frames) / 1e9,
            bytes as f64 * 8.0 / f64::from(seconds) / 1e6,
            f64::from(frames) / took
        );
    }
}
