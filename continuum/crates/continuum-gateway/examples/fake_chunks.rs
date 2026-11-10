//! Writes a directory of chunk files holding real JPEG frames of a moving square, so the gateway and the player can
//! be tried before any camera writes Tier 1: `cargo run -p continuum-gateway --example fake_chunks -- <dir> [seconds]`.
//! Stream 1, 15 frames per second, 2 second chunks, starting at the current time minus the length.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use continuum_core::StreamId;
use continuum_tier::{ChunkFormat, ChunkWriter};

fn jpeg(i: u32) -> Vec<u8> {
    let (w, h) = (320u16, 180u16);
    let mut rgb = vec![30u8; usize::from(w) * usize::from(h) * 3];
    let x = (i * 7) as usize % (usize::from(w) - 40);
    let y = 70 + ((i as f64 / 10.0).sin() * 50.0) as usize;
    for yy in y..y + 40 {
        for xx in x..x + 40 {
            let at = (yy * usize::from(w) + xx) * 3;
            rgb[at] = 230;
            rgb[at + 1] = 120 + (i % 100) as u8;
            rgb[at + 2] = 40;
        }
    }
    let mut out = Vec::new();
    jpeg_encoder::Encoder::new(&mut out, 80).encode(&rgb, w, h, jpeg_encoder::ColorType::Rgb).expect("encode");
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(dir) = args.get(1) else {
        eprintln!("usage: fake_chunks <dir> [seconds]");
        std::process::exit(2);
    };
    let seconds: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
    let dir = Path::new(dir);
    std::fs::create_dir_all(dir).expect("mkdir");
    let now_ns = SystemTime::now().duration_since(UNIX_EPOCH).expect("time").as_nanos() as u64;
    let start = now_ns - seconds * 1_000_000_000;
    let frame_ns = 1_000_000_000 / 15;
    let mut writer: Option<ChunkWriter> = None;
    let mut chunk_start = start;
    let mut n = 0u64;
    for i in 0..(seconds * 15) {
        let t = start + i * frame_ns;
        if writer.as_ref().is_some_and(|_| t - chunk_start >= 2_000_000_000) {
            writer.take().expect("writer").close().expect("close");
            n += 1;
        }
        if writer.is_none() {
            chunk_start = t;
            let path = dir.join(format!("1-{t}-{n}.cnk"));
            writer = Some(ChunkWriter::create(&path, StreamId(1), ChunkFormat::Passthrough, t).expect("create"));
        }
        writer.as_mut().expect("writer").append(t, t, i as u32, &jpeg(i as u32)).expect("append");
    }
    if let Some(w) = writer {
        w.close().expect("close");
    }
    println!("wrote {} seconds of stream 1 ({} frames) under {}", seconds, seconds * 15, dir.display());
}
