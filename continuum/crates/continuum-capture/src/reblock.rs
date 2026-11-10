//! Cutting a driver's chunks (PipeWire's quantum, a CoreAudio buffer) into the stream's fixed blocks, each placed
//! on the host clock from the chunk it started in.

/// A block cut from the incoming chunks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    /// Exactly the stream's block size.
    pub bytes: Vec<u8>,
    /// When its first frame was captured, host monotonic nanoseconds.
    pub captured_ns: u64,
}

/// Collects interleaved PCM frames and hands out blocks of `frames_per_block` frames.
#[derive(Clone, Debug)]
pub struct Reblocker {
    frame_bytes: usize,
    block_frames: usize,
    rate_hz: u32,
    pending: Vec<u8>,
    /// The capture time of `pending`'s first frame.
    pending_ns: u64,
}

impl Reblocker {
    /// A reblocker for frames of `frame_bytes` (channels times bytes per sample), blocks of `block_frames` frames,
    /// at `rate_hz`.
    #[must_use]
    pub fn new(frame_bytes: usize, block_frames: usize, rate_hz: u32) -> Self {
        Self {
            frame_bytes: frame_bytes.max(1),
            block_frames: block_frames.max(1),
            rate_hz,
            pending: Vec::new(),
            pending_ns: 0,
        }
    }

    /// Bytes per frame.
    #[must_use]
    pub fn frame_bytes(&self) -> usize {
        self.frame_bytes
    }

    /// Frames per block.
    #[must_use]
    pub fn block_frames(&self) -> usize {
        self.block_frames
    }

    /// Frames per second.
    #[must_use]
    pub fn rate_hz(&self) -> u32 {
        self.rate_hz
    }

    /// Bytes per block.
    #[must_use]
    pub fn block_bytes(&self) -> usize {
        self.frame_bytes * self.block_frames
    }

    /// Adds a chunk whose first frame was captured at `captured_ns`, returning every block now complete. A chunk
    /// with a torn last frame keeps the whole frames only. A chunk that doesn't follow on from what's pending (its
    /// time is more than half a block away from where the pending frames end) drops the pending frames, since they
    /// can't make a block that is continuous in time: the stream's clock then sees the gap.
    pub fn push(&mut self, chunk: &[u8], captured_ns: u64) -> Vec<Block> {
        let whole = chunk.len() - chunk.len() % self.frame_bytes;
        let chunk = &chunk[..whole];
        if self.pending.is_empty() {
            self.pending_ns = captured_ns;
        } else {
            let pending_frames = (self.pending.len() / self.frame_bytes) as u64;
            let expected = self.pending_ns + self.frames_ns(pending_frames);
            if expected.abs_diff(captured_ns) > self.frames_ns(self.block_frames as u64) / 2 {
                self.pending.clear();
                self.pending_ns = captured_ns;
            }
        }
        self.pending.extend_from_slice(chunk);
        let mut out = Vec::new();
        let size = self.block_bytes();
        while self.pending.len() >= size {
            let rest = self.pending.split_off(size);
            let bytes = std::mem::replace(&mut self.pending, rest);
            out.push(Block { bytes, captured_ns: self.pending_ns });
            self.pending_ns += self.frames_ns(self.block_frames as u64);
        }
        out
    }

    fn frames_ns(&self, frames: u64) -> u64 {
        if self.rate_hz == 0 { 0 } else { frames * 1_000_000_000 / u64::from(self.rate_hz) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quanta_become_fixed_blocks_placed_by_the_chunk_they_started_in() {
        // 48 kHz mono S16: 480-frame blocks from 1024-frame quanta.
        let mut r = Reblocker::new(2, 480, 48_000);
        let quantum = vec![1u8; 1_024 * 2];
        let a = r.push(&quantum, 1_000_000_000);
        assert_eq!(a.len(), 2);
        assert_eq!((a[0].captured_ns, a[1].captured_ns), (1_000_000_000, 1_010_000_000));
        let b = r.push(&quantum, 1_000_000_000 + 1_024 * 1_000_000_000 / 48_000);
        assert_eq!(b.len(), 2, "64 left over plus 1024 makes two more, 128 left");
        assert_eq!(b[0].captured_ns, 1_020_000_000, "continuous across chunks");
        assert!(b.iter().all(|blk| blk.bytes.len() == 960));
    }

    #[test]
    fn a_chunk_that_does_not_follow_on_drops_the_pending_frames() {
        let mut r = Reblocker::new(2, 480, 48_000);
        r.push(&[0u8; 100 * 2], 0);
        let out = r.push(&[0u8; 480 * 2], 1_000_000_000);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].captured_ns, 1_000_000_000, "a fresh start at the new chunk");
    }

    #[test]
    fn a_chunk_exactly_half_a_block_off_still_follows_on() {
        // 480-frame blocks at 48 kHz: half a block is 5 ms. 100 frames pending end at 100/48000 s.
        let mut r = Reblocker::new(2, 480, 48_000);
        r.push(&[0u8; 100 * 2], 0);
        let ends = 100 * 1_000_000_000 / 48_000;
        let out = r.push(&[0u8; 380 * 2], ends + 5_000_000);
        assert_eq!((out.len(), out[0].captured_ns), (1, 0), "at exactly half a block, it's jitter");
        let mut r = Reblocker::new(2, 480, 48_000);
        r.push(&[0u8; 100 * 2], 0);
        let out = r.push(&[0u8; 480 * 2], ends + 5_000_001);
        assert_eq!(out[0].captured_ns, ends + 5_000_001, "past it, a fresh start");
    }

    #[test]
    fn a_torn_frame_is_dropped() {
        let mut r = Reblocker::new(4, 2, 1_000);
        assert_eq!(r.push(&[0u8; 9], 0).len(), 1, "two whole 4-byte frames, one torn byte dropped");
    }
}
