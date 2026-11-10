//! A microphone made from a file, for tests: plays raw interleaved S16LE PCM through capture's real microphone path
//! (the stream registration, the broker-carved ring, the per-block stamps, and the per-batch `IngestNotice`s), in real
//! time, as a device would. Readers can't tell it from a microphone, which is the point: the audio worker's tests
//! (#548) read it live.
//!
//! ```bash
//! pcm_file_mic --socket data/state/continuum/broker.sock --name testmic --rate 16000 --channels 1 speech.s16
//! ```
//!
//! It registers `<name>.audio` and exits once the file is played (plus `--tail-ms` of silence, so a reader's gate
//! sees the end of the speech), printing `playing` when it starts.

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use continuum_capture::audio::{Chunk, ChunkSource, Microphone};
use continuum_capture::clock::{Clocks, SystemClocks};
use continuum_capture::manifest::Manifest;
use continuum_capture::register::PcmFormat;
use continuum_capture::worker::{WorkerEnd, microphone};
use continuum_client::{Client, ClientOptions};

/// The file, in 10 ms chunks paced to the wall clock, then silence, then the end.
struct FileMic {
    bytes: Vec<u8>,
    at: usize,
    chunk_bytes: usize,
    started: Instant,
    /// The host monotonic time the first chunk is stamped with; each next one 10 ms later, as a device's own clock
    /// would stamp them (stamping each when it's sent would carry the sleep's jitter, which capture's clock fit reads
    /// as lost samples).
    started_ns: u64,
    sent: u32,
    tail_chunks: u32,
}

impl ChunkSource for FileMic {
    fn next_chunk(&mut self, _timeout_ms: u32) -> std::io::Result<Option<Chunk>> {
        let due = self.started + Duration::from_millis(10 * u64::from(self.sent));
        if let Some(wait) = due.checked_duration_since(Instant::now()) {
            std::thread::sleep(wait);
        }
        let bytes = if self.at < self.bytes.len() {
            let end = (self.at + self.chunk_bytes).min(self.bytes.len());
            let mut b = self.bytes[self.at..end].to_vec();
            b.resize(self.chunk_bytes, 0);
            self.at = end;
            b
        } else if self.tail_chunks > 0 {
            self.tail_chunks -= 1;
            vec![0; self.chunk_bytes]
        } else {
            return Err(std::io::Error::other("the file is played"));
        };
        let captured_ns = self.started_ns + 10_000_000 * u64::from(self.sent);
        self.sent += 1;
        Ok(Some(Chunk { bytes, captured_ns }))
    }
}

impl Microphone for FileMic {
    fn dropped(&self) -> u64 {
        0
    }
}

struct Args {
    socket: PathBuf,
    name: String,
    rate: u32,
    channels: u16,
    tail_ms: u32,
    file: PathBuf,
}

fn parse() -> Result<Args, String> {
    let mut socket = None;
    let mut name = None;
    let mut rate = None;
    let mut channels = 1;
    let mut tail_ms = 500;
    let mut file = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{a} needs a value"));
        match a.as_str() {
            "--socket" => socket = Some(PathBuf::from(value()?)),
            "--name" => name = Some(value()?),
            "--rate" => rate = Some(value()?.parse().map_err(|e| format!("--rate: {e}"))?),
            "--channels" => channels = value()?.parse().map_err(|e| format!("--channels: {e}"))?,
            "--tail-ms" => tail_ms = value()?.parse().map_err(|e| format!("--tail-ms: {e}"))?,
            f if !f.starts_with("--") && file.is_none() => file = Some(PathBuf::from(f)),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(Args {
        socket: socket.ok_or("--socket is required")?,
        name: name.ok_or("--name is required")?,
        rate: rate.ok_or("--rate is required")?,
        channels,
        tail_ms,
        file: file.ok_or("a PCM file is required")?,
    })
}

fn main() -> std::process::ExitCode {
    let args = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("pcm_file_mic: {e}");
            eprintln!("usage: pcm_file_mic --socket S --name N --rate HZ [--channels C] [--tail-ms MS] FILE.s16");
            return std::process::ExitCode::from(2);
        }
    };
    let bytes = match std::fs::read(&args.file) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("pcm_file_mic: {}: {e}", args.file.display());
            return std::process::ExitCode::from(2);
        }
    };
    let manifest =
        format!("[[device]]\nname = \"{}\"\nkind = \"microphone\"\nmatch = {{ name = \"file\" }}\n", args.name);
    let device = Manifest::parse(&manifest).expect("a one-device manifest parses").devices.remove(0);
    let pcm = PcmFormat { rate_hz: args.rate, channels: args.channels, bytes_per_sample: 2, float: false };
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(1).enable_all().build().expect("a runtime");
    let mut client = match rt.block_on(Client::connect(&args.socket, ClientOptions::new("pcm_file_mic"))) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("pcm_file_mic: can't reach the broker: {e}");
            return std::process::ExitCode::from(1);
        }
    };
    let clocks = SystemClocks::new().expect("a boot id");
    let frame = usize::from(args.channels) * 2;
    let mut mic = FileMic {
        bytes,
        at: 0,
        chunk_bytes: frame * (args.rate as usize / 100),
        started: Instant::now(),
        started_ns: clocks.monotonic_ns(),
        sent: 0,
        tail_chunks: args.tail_ms / 10,
    };
    println!("playing");
    let _ = std::io::stdout().flush();
    let stop = AtomicBool::new(false);
    match microphone(rt.handle(), &mut client, &device, &mut mic, pcm, &clocks, &stop) {
        // The file ending is the source "failing": that's done.
        WorkerEnd::Device => std::process::ExitCode::SUCCESS,
        other => {
            eprintln!("pcm_file_mic: ended with {other:?}");
            std::process::ExitCode::from(1)
        }
    }
}
