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

    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut out = Out::new().u32(self.version).u32(u32::try_from(self.frames.len()).unwrap_or(u32::MAX));
        for f in &self.frames {
            out = out.u64(f.device_seq).u32(f.epoch).u8(f.flags());
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
        for _ in 0..count {
            let (device_seq, epoch, flags) = (i.u64()?, i.u32()?, i.u8()?);
            frames.push(FrameProvenance::from_wire(device_seq, epoch, flags).map_err(|_| bad("unknown flags"))?);
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
}
