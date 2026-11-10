//! The tier worker's library (docs/CONTINUUM.md, "Storage tiers"; issue #552).
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
pub mod v1audio;
pub mod v1video;
#[cfg(feature = "ffmpeg")]
pub mod video;
pub mod worker;

use continuum_core::StreamId;

pub use chunk::{ChunkError, ChunkFormat, ChunkReader, ChunkWriter, Record, scan_tail};
pub use flush::{ChunkCommit, Closed, FlushSettings, Flusher, Incoming, StreamShape};
pub use hydrate::{Hydrated, Picture, decode_jpeg, hydrate, hydrate_frame, read_chunk, rgb_to_nv12};
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
