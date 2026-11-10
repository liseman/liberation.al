//! A microphone worker's capture loop, over a [`ChunkSource`] so it's tested on synthetic PCM: the driver's chunks
//! (PipeWire quanta, CoreAudio buffers) are cut into the stream's fixed blocks, placed on the host clock, and
//! published, until the microphone stops, stalls, goes away, or the host wakes from sleep. The PipeWire source (and
//! CoreAudio's, to come) runs its callbacks on their own threads and hand chunks over through a [`ChunkSender`].

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel};
use std::time::Duration;

use crate::clock::{Anchor, Clocks, WakeWatch, take_anchor};
use crate::publish::{AudioStream, PublishError, Sink};
use crate::reblock::Reblocker;

/// A run of interleaved PCM frames and when its first frame was captured (host monotonic nanoseconds).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunk {
    /// Interleaved frames in the stream's format.
    pub bytes: Vec<u8>,
    /// When its first frame was captured.
    pub captured_ns: u64,
}

/// Where PCM comes from: a PipeWire or CoreAudio stream, or a script in tests.
pub trait ChunkSource {
    /// Waits up to `timeout_ms` for the next chunk. `Ok(None)` on a timeout.
    ///
    /// # Errors
    ///
    /// The device's error (it went away, or the stream broke).
    fn next_chunk(&mut self, timeout_ms: u32) -> std::io::Result<Option<Chunk>>;
}

/// A microphone a worker runs: a chunk source that can say how many chunks it had to drop.
pub trait Microphone: ChunkSource {
    /// Chunks dropped because the worker fell behind.
    fn dropped(&self) -> u64;
}

impl Microphone for ChannelSource {
    fn dropped(&self) -> u64 {
        ChannelSource::dropped(self)
    }
}

/// The callback thread's end: it never blocks, and a chunk the worker hasn't room for is counted and dropped (the
/// audio clock then sees the hole as a gap).
#[derive(Clone, Debug)]
pub struct ChunkSender {
    tx: SyncSender<Result<Chunk, String>>,
    dropped: Arc<AtomicU64>,
}

impl ChunkSender {
    /// Hands a chunk to the worker.
    pub fn send(&self, chunk: Chunk) {
        if let Err(TrySendError::Full(_)) = self.tx.try_send(Ok(chunk)) {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Tells the worker the stream failed.
    pub fn fail(&self, why: String) {
        let _ = self.tx.try_send(Err(why));
    }
}

/// The worker's end of the channel.
#[derive(Debug)]
pub struct ChannelSource {
    rx: Receiver<Result<Chunk, String>>,
    dropped: Arc<AtomicU64>,
}

/// A channel holding up to `depth` chunks.
#[must_use]
pub fn chunk_channel(depth: usize) -> (ChunkSender, ChannelSource) {
    let (tx, rx) = sync_channel(depth.max(1));
    let dropped = Arc::new(AtomicU64::new(0));
    (ChunkSender { tx, dropped: Arc::clone(&dropped) }, ChannelSource { rx, dropped })
}

impl ChannelSource {
    /// Chunks the callback had to drop because the worker fell behind.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

impl ChunkSource for ChannelSource {
    fn next_chunk(&mut self, timeout_ms: u32) -> std::io::Result<Option<Chunk>> {
        match self.rx.recv_timeout(Duration::from_millis(u64::from(timeout_ms))) {
            Ok(Ok(chunk)) => Ok(Some(chunk)),
            Ok(Err(why)) => Err(std::io::Error::other(why)),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err(std::io::Error::other("the audio stream ended")),
        }
    }
}

/// Why the loop returned.
#[derive(Debug)]
pub enum MicExit {
    /// `stop` was set.
    Stopped,
    /// No audio for the stall time.
    Stalled,
    /// The host slept and woke: the stream has re-anchored on this anchor, and the worker registers it again.
    Woke(Anchor),
    /// The device failed or went away.
    Source(std::io::Error),
    /// Publishing failed.
    Publish(PublishError),
}

/// Runs a microphone until it stops, stalls, fails, or the host wakes. The batch in progress is sent on the way
/// out, so the catalog has everything that reached the ring.
/// `health` is called with the stream every [`crate::video::HEALTH_NS`], for the worker's log (its counts and
/// fitted drift).
#[allow(clippy::too_many_arguments)]
pub fn run_microphone(
    source: &mut dyn ChunkSource,
    stream: &mut AudioStream,
    reblocker: &mut Reblocker,
    sink: &mut dyn Sink,
    clocks: &dyn Clocks,
    stall_ns: u64,
    stop: &AtomicBool,
    health: &mut dyn FnMut(&AudioStream),
) -> MicExit {
    let mut wake = WakeWatch::new(clocks);
    let mut last_health = clocks.monotonic_ns();
    let mut last = clocks.monotonic_ns();
    let step_ms = u32::try_from((stall_ns / 1_000_000).clamp(1, 100)).unwrap_or(100);
    let exit = loop {
        if stop.load(Ordering::Relaxed) {
            break MicExit::Stopped;
        }
        let now = clocks.monotonic_ns();
        if now.saturating_sub(last_health) >= crate::video::HEALTH_NS {
            last_health = now;
            health(stream);
        }
        if wake.woke(clocks) {
            let anchor = take_anchor(clocks);
            *reblocker = Reblocker::new(reblocker.frame_bytes(), reblocker.block_frames(), reblocker.rate_hz());
            if let Err(e) = stream.rewake(anchor, sink) {
                break MicExit::Publish(e);
            }
            return MicExit::Woke(anchor);
        }
        match source.next_chunk(step_ms) {
            Ok(Some(chunk)) => {
                let arrival = clocks.monotonic_ns();
                last = arrival;
                for block in reblocker.push(&chunk.bytes, chunk.captured_ns) {
                    if let Err(e) = stream.block(&block.bytes, block.captured_ns, arrival, sink) {
                        return MicExit::Publish(e);
                    }
                }
            }
            Ok(None) => {
                if clocks.monotonic_ns().saturating_sub(last) >= stall_ns {
                    break MicExit::Stalled;
                }
            }
            // A signal that interrupts a wait is the stop it carries.
            Err(_) if stop.load(Ordering::Relaxed) => break MicExit::Stopped,
            Err(e) => break MicExit::Source(e),
        }
    };
    match stream.flush(sink) {
        Ok(()) => exit,
        Err(e) => MicExit::Publish(e),
    }
}
