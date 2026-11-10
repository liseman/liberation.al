//! Writes a time range of one video stream's chunks as an ordinary MP4, for a player or ffprobe (#553).
//!
//! ```text
//! continuum-export --out clip.mp4 [--from-ns N] [--to-ns N] [--codec auto|copy|h264] [--force] CHUNK...
//! ```
//!
//! The chunks are one stream's, in time order (the catalog's `CHUNKS` rows name them). `auto`, the default, copies a
//! raw camera's H.264 (rounded out to whole GOPs) and encodes H.264 otherwise (exactly the range). Mixed dimensions use the largest selected frame's canvas with aspect-preserving black bars;
//! explicit `copy` rejects that boundary. `h264` always re-encodes. The range defaults to everything in the chunks.
//! An existing file at `--out` is replaced only with `--force`.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use continuum_tier::chunk::ChunkReader;
use continuum_tier::export::{ExportCodec, export};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Codec {
    Auto,
    Copy,
    H264,
}

/// Writes a time range of one video stream's chunks as an ordinary MP4.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// The MP4 to write.
    #[arg(long)]
    out: PathBuf,
    /// Start of the range, UTC nanoseconds (default: the earliest frame).
    #[arg(long)]
    from_ns: Option<u64>,
    /// End of the range, exclusive, UTC nanoseconds (default: after the last frame).
    #[arg(long)]
    to_ns: Option<u64>,
    /// How to write the video.
    #[arg(long, value_enum, default_value_t = Codec::Auto)]
    codec: Codec,
    /// Replace --out if it exists.
    #[arg(long)]
    force: bool,
    /// One stream's chunk files, in time order.
    #[arg(required = true)]
    chunks: Vec<PathBuf>,
}

fn run(args: &Args) -> Result<String, String> {
    if args.out.exists() && !args.force {
        return Err(format!("{} exists; pass --force to replace it", args.out.display()));
    }
    let (mut first, mut last) = (u64::MAX, 0u64);
    for path in &args.chunks {
        let reader = ChunkReader::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        for r in reader.records() {
            first = first.min(r.start_ns);
            last = last.max(r.end_exclusive());
        }
    }
    let codec = match args.codec {
        Codec::Copy => ExportCodec::Copy,
        Codec::H264 => ExportCodec::H264,
        Codec::Auto => ExportCodec::Auto,
    };
    let (from, to) = (args.from_ns.unwrap_or(first), args.to_ns.unwrap_or(last));
    let chunks: Vec<&std::path::Path> = args.chunks.iter().map(PathBuf::as_path).collect();
    let done = export(&chunks, &args.out, from, to, codec).map_err(|e| e.to_string())?;
    Ok(format!("wrote {}: {} frames of {}x{} h264", args.out.display(), done.frames, done.size.0, done.size.1))
}

fn main() -> ExitCode {
    continuum_log::init("continuum-export");
    let args = Args::parse();
    match run(&args) {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("continuum-export: {e}");
            ExitCode::FAILURE
        }
    }
}
