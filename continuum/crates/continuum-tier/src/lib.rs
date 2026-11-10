//! The tier worker's library (docs/CONTINUUM.md, "Storage tiers").
//!
//! The tier worker drains Tier 0 into frame-indexed Tier 1 chunks before slots are reclaimed, and carries out purges
//! on its files. A microphone's chunk becomes Opus ([`opus`]) as it closes, before it is committed. This crate is the part of it that needs no
//! socket: the chunk container ([`chunk`]), the payload codecs ([`codecs`]), the flusher that cuts chunks on time
//! ([`flush`]), reading frames back ([`hydrate`]), startup recovery of chunks a crash left open ([`recover`]), and
//! the codec probe ([`probe`]). The worker program that connects to the broker and follows streams by subscription
//! is [`service`].
//!
//! Containers: payloads that are already compressed or self-contained (MJPEG frames, FLAC, LZ4 raw frames, Arrow
//! IPC) go into the crate's own append-only chunk file, which is crash safe by construction and carries a frame
//! index. A raw camera's H.264 access units and Opus audio go in the same container, laid out by [`gop`].
//!
//! Behind the `ffmpeg` feature: [`h264`] encodes a raw camera's Tier 1 H.264, [`video`] decodes video, and
//! [`tier2`] reads frames out of video chunks. [`export`] (and the `continuum-export` command) writes a range as an
//! ordinary MP4. Nothing recompresses chunks after they are committed: video stays in Tier 1 until it ages out.

/// Every line this crate writes to stderr, stamped with the local date, time, and UTC offset when its service turned
/// stamping on (`log::stamp_lines`), as the broker's log lines are. Shadows the standard macro crate-wide.
#[macro_export]
macro_rules! eprintln {
    ($($arg:tt)*) => {
        ::std::eprintln!("{}{}", $crate::log::prefix(), ::std::format_args!($($arg)*))
    };
}

/// Stderr line stamps.
pub mod log {
    use std::sync::atomic::{AtomicBool, Ordering};

    static STAMP: AtomicBool = AtomicBool::new(false);

    /// Stamp every later stderr line. Services call it first thing; command-line tools leave their output as it is.
    pub fn stamp_lines() {
        STAMP.store(true, Ordering::Relaxed);
    }

    /// The stamp a line starts with: `2026-11-03T02:20:15-08:00 ` once stamping is on (the broker's format), else
    /// nothing.
    #[must_use]
    pub fn prefix() -> String {
        if STAMP.load(Ordering::Relaxed) { format!("{} ", stamp()) } else { String::new() }
    }

    /// Local date and time to the second, with its UTC offset.
    #[must_use]
    pub fn stamp() -> String {
        chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%:z").to_string()
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn a_stamp_is_the_local_iso_time_with_its_offset() {
            let stamp = super::stamp();
            let shape: String = stamp.chars().map(|c| if c.is_ascii_digit() { 'd' } else { c }).collect();
            assert!(["dddd-dd-ddTdd:dd:dd-dd:dd", "dddd-dd-ddTdd:dd:dd+dd:dd"].contains(&shape.as_str()), "{stamp}");
        }

        #[test]
        fn lines_are_unstamped_until_a_service_turns_stamping_on() {
            // One test, since the switch is process-wide and only ever turns on.
            assert_eq!(super::prefix(), "");
            super::stamp_lines();
            let prefix = super::prefix();
            assert!(prefix.ends_with(' ') && prefix.len() == 26, "{prefix:?}");
        }
    }
}

pub mod archive;
#[cfg(feature = "ffmpeg")]
pub mod av1;
pub mod chunk;
pub mod codecs;
mod copies;
#[cfg(feature = "ffmpeg")]
mod encode;
#[cfg(feature = "ffmpeg")]
pub mod export;
pub mod flush;
pub mod gop;
#[cfg(feature = "ffmpeg")]
pub mod h264;
pub mod hydrate;
pub mod inplace_audio;
pub mod inplace_video;
pub mod layout;
pub mod mp4;
pub mod opus;
pub mod pcm;
pub mod probe;
pub mod raw;
pub mod recent;
pub mod recording_bound;
#[cfg(test)]
mod recording_bound_tests;
pub mod recover;
pub mod service;
pub mod sweep;
#[cfg(feature = "ffmpeg")]
pub mod tier2;
#[cfg(feature = "ffmpeg")]
pub mod video;
pub mod worker;

use continuum_core::StreamId;

pub use chunk::{ChunkError, ChunkFormat, ChunkReader, ChunkWriter, Record, scan_tail};
pub use flush::{ChunkCommit, Closed, FlushSettings, Flusher, Incoming, StreamShape};
pub use hydrate::{Hydrated, JpegNv12, Picture, decode_jpeg, hydrate, hydrate_frame, read_chunk, rgb_to_nv12};
pub use probe::{Ffmpeg, Probe, probe};
pub use recover::{Recovered, recover};
pub use worker::{Appended, CameraFrameInfo, CommitFailure, Counts, Outputs, Payload, WorkerLoop};

/// What the tier library can fail with.
#[derive(Debug, thiserror::Error)]
pub enum TierError {
    /// A chunk file problem.
    #[error(transparent)]
    Chunk(#[from] ChunkError),
    /// A codec refused its input.
    #[error("{0}")]
    Codec(String),
    /// A payload for a stream the flusher wasn't told about.
    #[error("stream {} was not declared to the flusher", .0.0)]
    UnknownStream(StreamId),
    /// A payload header couldn't be built.
    #[error(transparent)]
    Header(#[from] continuum_core::DecodeError),
    /// A frame a purge cut from its chunk: the place is kept, the bytes are gone.
    #[error("frame {0} was purged from the chunk")]
    Purged(usize),
    /// Something that needs the `ffmpeg` feature, in a build without it.
    #[error("{0} needs the ffmpeg feature")]
    Unsupported(&'static str),
    /// The file system failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
