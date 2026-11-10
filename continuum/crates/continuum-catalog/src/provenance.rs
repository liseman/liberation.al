//! Per-frame capture provenance of a chunk (schema minor 27, catalog format 4): the device sequence and capture
//! epoch each absorbed Tier 0 header carried, kept after the headers are compacted away. A rewrite or aging keeps
//! every frame in its place, so the row never changes after the commit. A chunk without one is unknown, never
//! continuous: nothing here is inferred from times or a frame rate.

use crate::codec::{In, Out};
use crate::{CatalogError, Result};

/// The only provenance layout so far.
pub const PROVENANCE_VERSION: u32 = 1;

const ARRIVAL: u8 = 1;

/// What capture attested about one frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameProvenance {
    /// The device's or driver's frame sequence.
    pub device_seq: u64,
    /// The capture session it came from: frames are comparable by `device_seq` only within one nonzero epoch.
    /// Zero means the producer didn't say.
    pub epoch: u32,
    /// Stamped when the frame arrived, not by the device.
    pub stamped_at_arrival: bool,
}

impl FrameProvenance {
    /// The wire flags byte.
    #[must_use]
    pub fn flags(&self) -> u8 {
        if self.stamped_at_arrival { ARRIVAL } else { 0 }
    }

    /// From the wire fields.
    ///
    /// # Errors
    /// [`CatalogError::Invalid`] for a flag bit this version doesn't define.
    pub fn from_wire(device_seq: u64, epoch: u32, flags: u8) -> Result<Self> {
        if flags & !ARRIVAL != 0 {
            return Err(CatalogError::Invalid(format!("unknown frame provenance flags {flags:#x}")));
        }
        Ok(Self { device_seq, epoch, stamped_at_arrival: flags & ARRIVAL != 0 })
    }
}

/// A chunk's provenance: one entry per frame, in the frame index's order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkProvenance {
    /// [`PROVENANCE_VERSION`].
    pub version: u32,
    /// One per frame.
    pub frames: Vec<FrameProvenance>,
}

impl ChunkProvenance {
    /// Version 1 provenance for these frames.
    #[must_use]
    pub fn new(frames: Vec<FrameProvenance>) -> Self {
        Self { version: PROVENANCE_VERSION, frames }
    }

    /// Checks it against the chunk it describes.
    ///
    /// # Errors
    /// [`CatalogError::Invalid`] for an unknown version or a count that isn't the chunk's frame count.
    pub fn validate(&self, frames: usize) -> Result<()> {
        if self.version != PROVENANCE_VERSION {
            return Err(CatalogError::Invalid(format!("unknown frame provenance version {}", self.version)));
        }
        if self.frames.len() != frames {
            return Err(CatalogError::Invalid(format!(
                "frame provenance has {} entries for {frames} frames",
                self.frames.len()
            )));
        }
        Ok(())
    }

    /// A chunk's provenance after a part of `added` frames joined its first `known` frames: one entry per frame of
    /// the whole chunk, the side that attested nothing reading as unknown (epoch 0). `None` when neither side did.
    #[must_use]
    pub fn joined(stored: Option<Self>, known: usize, part: Option<&Self>, added: usize) -> Option<Self> {
        if stored.is_none() && part.is_none() {
            return None;
        }
        let mut frames = stored.map_or_else(Vec::new, |p| p.frames);
        frames.resize(known, FrameProvenance::default());
        match part {
            Some(p) => frames.extend_from_slice(&p.frames),
            None => frames.resize(known + added, FrameProvenance::default()),
        }
        Some(Self::new(frames))
    }

    /// Runs of frames in one epoch with one flags byte and consecutive device sequences, which is what a camera
    /// that drops nothing sends: a run is its length, its first device sequence, its epoch, and its flags.
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut out = Out::new().u32(self.version).u32(u32::try_from(self.frames.len()).unwrap_or(u32::MAX));
        let mut at = 0;
        while at < self.frames.len() {
            let first = self.frames[at];
            let mut len = 1;
            while self.frames.get(at + len).is_some_and(|f| {
                f.epoch == first.epoch
                    && f.flags() == first.flags()
                    && f.device_seq == first.device_seq.wrapping_add(len as u64)
            }) {
                len += 1;
            }
            out = out.var(len as u64).var(first.device_seq).var(u64::from(first.epoch)).u8(first.flags());
            at += len;
        }
        out.done()
    }

    pub(crate) fn decode(b: &[u8]) -> Result<Self> {
        let bad = |why: &str| CatalogError::Storage(format!("frame provenance: {why}").into());
        let mut i = In::new(b);
        let version = i.u32()?;
        if version != PROVENANCE_VERSION {
            return Err(bad("unknown version"));
        }
        let count = i.u32()? as usize;
        if count > crate::MAX_CHUNK_FRAMES {
            return Err(bad("too many frames"));
        }
        let mut frames = Vec::with_capacity(count);
        while frames.len() < count {
            let (len, device_seq, epoch, flags) = (i.var()?, i.var()?, i.var()?, i.u8()?);
            let len = usize::try_from(len).ok().filter(|&n| n > 0 && n <= count - frames.len());
            let (Some(len), Ok(epoch)) = (len, u32::try_from(epoch)) else { return Err(bad("a run doesn't fit")) };
            let first = FrameProvenance::from_wire(device_seq, epoch, flags).map_err(|_| bad("unknown flags"))?;
            frames.extend((0..len).map(|n| FrameProvenance { device_seq: device_seq.wrapping_add(n as u64), ..first }));
        }
        if !i.is_empty() {
            return Err(bad("trailing bytes"));
        }
        Ok(Self { version, frames })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provenance_round_trips_and_refuses_damage() {
        let p = ChunkProvenance::new(vec![
            FrameProvenance { device_seq: 7, epoch: 0xdead_beef, stamped_at_arrival: false },
            FrameProvenance { device_seq: u64::MAX, epoch: 0, stamped_at_arrival: true },
        ]);
        let bytes = p.encode();
        assert_eq!(ChunkProvenance::decode(&bytes).unwrap(), p);
        assert!(ChunkProvenance::decode(&bytes[..bytes.len() - 1]).is_err(), "truncated");
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(ChunkProvenance::decode(&extra).is_err(), "trailing");
        let mut flags = bytes.clone();
        *flags.last_mut().unwrap() = 2;
        assert!(ChunkProvenance::decode(&flags).is_err(), "undefined flag bit");
        let mut version = bytes;
        version[..4].copy_from_slice(&2u32.to_be_bytes());
        assert!(ChunkProvenance::decode(&version).is_err(), "unknown version");
    }

    #[test]
    fn a_camera_that_drops_nothing_is_one_run() {
        let steady = ChunkProvenance::new(
            (0..1800)
                .map(|n| FrameProvenance { device_seq: 40_000 + n, epoch: 7, stamped_at_arrival: false })
                .collect(),
        );
        let bytes = steady.encode();
        assert_eq!(ChunkProvenance::decode(&bytes).unwrap(), steady);
        assert!(bytes.len() < 20, "{} bytes", bytes.len());
        let mut long = bytes.clone();
        long[8] = 0xff; // the run claims more frames than the count
        assert!(ChunkProvenance::decode(&long).is_err());
    }

    proptest::proptest! {
        #[test]
        fn any_provenance_round_trips(
            rows in proptest::collection::vec((0u64..4, 0u32..3, proptest::bool::ANY), 0..64)
        ) {
            let mut seq = 0u64;
            let p = ChunkProvenance::new(
                rows.into_iter()
                    .map(|(step, epoch, stamped_at_arrival)| {
                        seq = seq.wrapping_add(step);
                        FrameProvenance { device_seq: seq, epoch, stamped_at_arrival }
                    })
                    .collect(),
            );
            proptest::prop_assert_eq!(ChunkProvenance::decode(&p.encode()).unwrap(), p);
        }
    }
}
