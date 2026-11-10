//! Continuum's Tier 0 data plane: a POSIX shared memory arena of 16 KB aligned slab slots and rings.
//!
//! See [`arena`] for the layout, the slot state machine, and how pins and leases work.

pub mod arena;
pub mod ring;
pub mod seqwait;
pub mod sys;

use std::io;
use std::time::Duration;

pub use arena::{
    Arena, ArenaConfig, DEFAULT_LEASE, Lease, MAX_CARVED, MAX_LEASE, MAX_PINS, Overwritten, Purged, Reclaimed,
    RingSpec, SlabClass, SlotScan, SlotState, WriteSlot,
};
pub use continuum_core::{PAGE_ALIGN, SlotHandle};
pub use ring::{CarvedRing, ReadFrom, Record, RingReader, RingWriter};
pub use seqwait::{SeqWait, deadline_after};

/// Everything that can go wrong in the arena.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// This writer's claim on the ring was taken away (the broker released it when the writer's session ended), so
    /// it may no longer write.
    #[error("this writer no longer holds the ring")]
    NotWriter,
    /// The shared memory name breaks the portable rules.
    #[error("shared memory name {0:?} must start with '/', have no other '/', and be at most 31 bytes")]
    BadName(String),
    /// The arena's size overflows.
    #[error("the arena is too large")]
    TooLarge,
    /// The arena exists but its creator hasn't finished setting it up.
    #[error("the arena isn't initialized yet")]
    NotReady,
    /// The shared memory object isn't a Continuum arena, or has an incompatible layout.
    #[error("not a Continuum arena, or an incompatible version")]
    BadArena,
    /// The arena config is invalid.
    #[error("bad arena config: {0}")]
    BadConfig(String),
    /// The slab class or slot index doesn't exist.
    #[error("no such slab class or slot")]
    NoSuchSlot,
    /// The ring doesn't exist.
    #[error("no such ring")]
    NoSuchRing,
    /// The handle's slot is being reclaimed, was reused, or isn't published.
    #[error("stale handle: the slot is being reclaimed, was reused, or isn't published")]
    Stale,
    /// Every slot in the class is in use.
    #[error("every slot in the class is in use")]
    Full,
    /// The slot already holds MAX_PINS pins.
    #[error("the slot has the most pins it can hold")]
    Busy,
    /// The lease expired, was released, or lost its slot to an eviction.
    #[error("the lease expired")]
    Expired,
    /// A lease length of zero or over the maximum.
    #[error("lease length {0:?} must be over 0 and at most MAX_LEASE")]
    BadLease(Duration),
    /// The payload doesn't fit the slot.
    #[error("a payload of {len} bytes doesn't fit a {capacity}-byte slot")]
    TooLong {
        /// The payload's length.
        len: usize,
        /// The slot's size.
        capacity: usize,
    },
    /// Another writer holds the ring.
    #[error("ring {0} already has a writer")]
    RingBusy(usize),
    /// A buffer of the wrong size for the payload.
    #[error("the buffer is {got} bytes; the payload is {expected}")]
    WrongLength {
        /// The payload's length.
        expected: usize,
        /// The buffer's length.
        got: usize,
    },
    /// A ring record of the wrong size.
    #[error("a record of {got} bytes doesn't match the ring's stride of {stride}")]
    BadRecord {
        /// The record's length.
        got: usize,
        /// The ring's stride.
        stride: usize,
    },
    /// An operating system error.
    #[error(transparent)]
    Io(#[from] io::Error),
}
