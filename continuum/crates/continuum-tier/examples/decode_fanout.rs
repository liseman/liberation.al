//! Bounded, read-only decode-component comparison, excluding IPC, model uploads, and color conversion.
//! Run from continuum at standard priority through the shared build slot:
//! `../tools/build-slot.sh cargo run --release -p continuum-tier --features ffmpeg --example decode_fanout -- --input /private/fixture.mp4 --gpu 0 --max-frames 300 --consumers 5`
//! Reads at most 128 MiB once, feeds at most 600 H.264 samples once per decoder, and drains EOF.
//! Software fanout uses N independent decoders; the shared CUDA mode uses one decoder. No services are contacted.

use std::collections::BTreeSet;
use std::error::Error;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use continuum_tier::mp4::{read_video, to_annex_b};
use continuum_tier::video::{H264DecodeBackend, H264StreamDecoder, TaggedFrame};
use serde_json::{Value, json};

const MAX_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Parser)]
struct Args {
    /// Immutable local MP4 fixture. Never included in measurement output.
    #[arg(long)]
    input: PathBuf,
    /// CUDA-visible ordinal; availability is checked by the decoder.
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u32).range(0..=2_147_483_647))]
    gpu: u32,
    /// Maximum access units, with no repeat loops.
    #[arg(long, default_value_t = 300, value_parser = clap::value_parser!(u32).range(1..=600))]
    max_frames: u32,
    /// Independent software consumers compared against a single shared CUDA decoder.
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u32).range(1..=8))]
    consumers: u32,
}

struct Unit {
    bytes: Vec<u8>,
    pts_ns: i64,
}

fn cpu_seconds() -> Result<f64, Box<dyn Error>> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage writes a complete rusage on success; the pointer is valid and properly aligned.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    // SAFETY: the successful getrusage call initialized this value.
    let usage = unsafe { usage.assume_init() };
    Ok(usage.ru_utime.tv_sec as f64
        + usage.ru_utime.tv_usec as f64 / 1_000_000.0
        + usage.ru_stime.tv_sec as f64
        + usage.ru_stime.tv_usec as f64 / 1_000_000.0)
}

fn collect(
    output: Vec<TaggedFrame>,
    seen: &mut BTreeSet<u64>,
    bytes: &mut u64,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn Error>> {
    for frame in output {
        if frame.width != width
            || frame.height != height
            || frame.bgr.is_some()
            || frame.nv12.len() as u64 != u64::from(width) * u64::from(height) * 3 / 2
            || !seen.insert(frame.tag)
        {
            return Err("decoded output changed geometry, duplicated identity, or violated NV12-only layout".into());
        }
        *bytes += frame.nv12.len() as u64;
    }
    Ok(())
}

fn measure(
    units: &[Unit],
    width: u32,
    height: u32,
    media_seconds: f64,
    consumers: u32,
    backend: H264DecodeBackend,
) -> Result<Value, Box<dyn Error>> {
    let count = if backend == H264DecodeBackend::Software { consumers } else { 1 };
    let cpu = cpu_seconds()?;
    let start = Instant::now();
    let mut decoders = (0..count).map(|_| H264StreamDecoder::with_backend(backend)).collect::<Result<Vec<_>, _>>()?;
    let init_wall = start.elapsed().as_secs_f64();
    let init_cpu = cpu_seconds()? - cpu;
    let expected: BTreeSet<_> = (1..=units.len() as u64).collect();
    let cpu = cpu_seconds()?;
    let start = Instant::now();
    let mut bytes = 0;
    for decoder in &mut decoders {
        let mut seen = BTreeSet::new();
        for (index, unit) in units.iter().enumerate() {
            collect(
                decoder.push_with_pts(&unit.bytes, index as u64 + 1, unit.pts_ns)?,
                &mut seen,
                &mut bytes,
                width,
                height,
            )?;
        }
        collect(decoder.finish()?, &mut seen, &mut bytes, width, height)?;
        if seen != expected {
            return Err("decoded output tags do not match every input after EOF drain".into());
        }
    }
    let wall = start.elapsed().as_secs_f64();
    let cpu = cpu_seconds()? - cpu;
    Ok(json!({"mode": if backend == H264DecodeBackend::Software {"independent_software"} else {"shared_cuda"},
        "scope": "decode_component_only_excludes_ipc_model_uploads_color_conversion",
        "media_seconds": media_seconds, "frames": units.len(), "width": width, "height": height,
        "consumer_count": consumers, "decoder_count": count, "init_cpu_seconds": init_cpu, "init_wall_seconds": init_wall,
        "cpu_seconds": cpu, "wall_seconds": wall, "steady_cpu_per_media_second": cpu / media_seconds,
        "bytes_output": bytes, "output": "host_nv12", "eof_tags_complete": true}))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let file = File::open(&args.input)?;
    if !file.metadata()?.is_file() || file.metadata()?.len() > MAX_BYTES {
        return Err("fixture must be a regular file no larger than 128 MiB".into());
    }
    let mut bytes = Vec::new();
    file.take(MAX_BYTES).read_to_end(&mut bytes)?;
    let video = read_video(&mut Cursor::new(&bytes))?;
    if !matches!(&video.codec, b"avc1" | b"avc3")
        || video.timescale == 0
        || video.samples.is_empty()
        || !video.samples[0].key
        || video.width == 0
        || video.height == 0
    {
        return Err("fixture must start with a keyframe and contain a valid H.264 video track".into());
    }
    let samples = &video.samples[..video.samples.len().min(args.max_frames as usize)];
    let first = samples.iter().map(|sample| sample.pts).min().ok_or("no samples")?;
    let last =
        samples.iter().map(|sample| i128::from(sample.pts) + i128::from(sample.duration)).max().ok_or("no samples")?;
    let media_seconds = (last - i128::from(first)) as f64 / f64::from(video.timescale);
    if media_seconds <= 0.0 {
        return Err("fixture has no positive presentation duration".into());
    }
    let mut units = Vec::with_capacity(samples.len());
    for sample in samples {
        let start = usize::try_from(sample.offset)?;
        let end = start.checked_add(sample.size as usize).ok_or("sample offset overflow")?;
        let raw = bytes.get(start..end).ok_or("sample exceeds bounded fixture bytes")?;
        // Preserve the actual edited MP4 presentation clock, including regressing B-frame input PTS.
        let pts_ns = i64::try_from(i128::from(sample.pts) * 1_000_000_000 / i128::from(video.timescale))?;
        units.push(Unit { bytes: to_annex_b(&video, raw, sample.key)?, pts_ns });
    }
    let software = measure(
        &units,
        u32::from(video.width),
        u32::from(video.height),
        media_seconds,
        args.consumers,
        H264DecodeBackend::Software,
    )?;
    let cuda = measure(
        &units,
        u32::from(video.width),
        u32::from(video.height),
        media_seconds,
        args.consumers,
        H264DecodeBackend::Cuda { gpu: args.gpu },
    )?;
    println!("{}", serde_json::to_string(&json!({"measurements": [software, cuda]}))?);
    Ok(())
}
