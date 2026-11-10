//! The arena: one shared memory object holding a superblock, the slot directory, slab slots, and ring segments.
//!
//! # Layout
//!
//! ```text
//! offset 0                 superblock: magic, layout version, size, arena generation, publish counter, policy
//!                          flags, and the class and ring descriptors, padded to a 16 KB page
//!                          slot directory: one 64-byte SlotMeta per slot; ring control blocks, 128 bytes each
//! next 16 KB boundary      class 0 slots, each slot_size bytes (a multiple of 16 KB), then class 1, ...
//!                          ring 0 records (stride * capacity, padded to a multiple of 16 KB), then ring 1, ...
//! ```
//!
//! Every slot and ring segment offset and length is a multiple of [`PAGE_ALIGN`], and so is the mapping's base
//! address (see `sys::map_aligned`). So a Metal worker can wrap a whole slot (its class size, not the payload
//! length) with `makeBuffer(bytesNoCopy:length:)` in `storageModeShared`, and a CUDA worker can `cudaHostRegister`
//! the whole mapping with the default flag. Registering faults in every page, so the first CUDA worker makes the
//! arena resident. The design doc (docs/CONTINUUM.md) puts the directory inside a 64 KB superblock; it has its own region here
//! from the start, so slot counts never outgrow it.
//!
//! # Slot states
//!
//! Each slot's state, generation, and pin count share one 64-bit word (`generation << 32 | pins << 8 | state`), so
//! one compare-and-swap moves all three together:
//!
//! ```text
//! Free(g) --claim--> Writing(g) --publish--> Ready(g) --reclaim--> Reclaiming(g) --pins gone--> Free(g + 1)
//!                    Writing(g) --dropped unpublished--> Free(g + 1)
//! ```
//!
//! The generation goes up every time a slot is freed, so a [`SlotHandle`] or [`Lease`] from before can't pin or
//! release the new contents (the ABA problem). It skips 0.
//!
//! # Pins and leases
//!
//! A reader pins a Ready slot with a lease: it raises the slot's lease expiry (the newest pin's deadline, on the
//! system-wide monotonic clock) to its own deadline, then increments the pin count. Releasing decrements the count,
//! and only for the generation the lease was taken on, so a release after the slot was reused does nothing.
//!
//! A Reclaiming slot is freed when its last pin goes. It's also freed while still pinned once every pin's lease has
//! expired, so a reader that stalls (a long inference, a GC pause) can't keep a producer from reusing it. If the
//! arena was created with `reclaim_requires_flush`, that only happens after the tier worker has marked the slot
//! flushed to Tier 1, so nothing is lost. The evicted reader finds out from [`Lease::is_valid`], which checks the
//! generation; readers check it after reading.
//!
//! A standing reader renews its lease before the deadline. A renew also bumps a counter in the state word, so an
//! eviction that read the old deadline fails its compare-and-swap and rereads it: a renew that returns Ok holds.
//!
//! The slot's expiry is the newest deadline of any lease taken on the generation, including released ones, so a
//! released lease still delays evicting a stalled reader by up to its ttl. That's why leases are capped at
//! [`MAX_LEASE`] (60 s, the default is 5 s) and standing readers renew rather than take one long lease.
//!
//! Slot bytes are plain memory: producers write them in place (a decoder, a DMA), and readers read them in place.
//! The lease protocol keeps the two apart, so the one race left is a reader still reading after its lease expired
//! and the slot was reused. [`Lease::copy_to`] and [`Lease::is_valid`] detect that after the fact; the reader
//! must not trust bytes it read past its deadline. (Rings, whose overwrites are routine, copy records atomically.)
//!
//! # Claiming
//!
//! [`Arena::claim`] takes a free slot. [`Arena::claim_overwriting`] also reclaims the oldest unpinned Ready slot
//! when none is free, so a producer never waits: that's the policy for a capture stream's own slot ring (a class
//! sized for that stream's frames, about 10 seconds of them by default). It overwrites whether or not the slot was
//! flushed; an unflushed overwrite is a Tier 1 gap, which it reports ([`Overwritten`]) so the broker can record it.
//! Live capture never stalls for the archive. The size classes for derived payloads use plain claims and explicit reclaims.
//! Each claim records an owner tag (a stream, lease, or worker id) so the broker can count and reclaim per owner.
//!
//! # Crashes
//!
//! A crashed reader's pins expire on their own. A producer that crashes mid-write leaves its slot in Writing, and a
//! crashed ring writer leaves its ring claimed; the broker tracks pins and claims per connection and recovers those
//! (Phase 2).

use std::mem::size_of;
use std::os::fd::BorrowedFd;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering, fence};
use std::time::Duration;

use continuum_core::{PAGE_ALIGN, SlotHandle, checked_align_up_16k, is_aligned_16k};

use crate::Error;
use crate::sys::{self, Mapping};

const MAGIC: u64 = u64::from_le_bytes(*b"CONTNUUM");
const VERSION: u32 = 4;

/// Most slab size classes one arena can have.
pub const MAX_CLASSES: usize = 16;
/// Most rings one arena can have.
pub const MAX_RINGS: usize = 32;
/// Most pins one slot can hold at once.
pub const MAX_PINS: u32 = (1 << 24) - 1;
/// The default lease length (the design doc's 5 seconds).
pub const DEFAULT_LEASE: Duration = Duration::from_secs(5);

const FREE: u64 = 0;
const WRITING: u64 = 1;
const READY: u64 = 2;
const RECLAIMING: u64 = 3;

const FLAG_RECLAIM_REQUIRES_FLUSH: u32 = 1;

/// One slab size class: `slot_count` slots of `slot_size` bytes. The size is rounded up to a multiple of
/// [`PAGE_ALIGN`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlabClass {
    /// Bytes per slot, before rounding up to a multiple of [`PAGE_ALIGN`].
    pub slot_size: usize,
    /// How many slots.
    pub slot_count: u32,
}

/// One ring: `capacity` fixed-size records of `stride` bytes, for streams like PCM audio, IMU samples, or slot
/// handles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RingSpec {
    /// Bytes per record. A multiple of 8 lets records be copied in 8-byte words; any other stride copies byte by
    /// byte, about 8 times the instructions (round PCM blocks up if it matters).
    pub stride: u32,
    /// How many records the ring keeps before overwriting the oldest.
    pub capacity: u32,
}

/// What an arena holds, and its reclaim policy.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ArenaConfig {
    /// Slab size classes, in order: class `i` is `classes[i]`.
    pub classes: Vec<SlabClass>,
    /// Rings, in order.
    pub rings: Vec<RingSpec>,
    /// Bumped each time the broker recreates the arena; part of its name and of every Tier 0 locator.
    pub generation: u32,
    /// Only evict a slot whose pins expired once it's flushed to Tier 1. Off until the tier worker exists (Phase
    /// 4); with it off, an expired pin never blocks reuse.
    pub reclaim_requires_flush: bool,
    /// Bytes reserved at the end of the arena for rings carved while it's live (see [`Arena::carve_ring`]), rounded
    /// up to a multiple of [`PAGE_ALIGN`]. 0 for none.
    pub ring_region_bytes: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct ClassDesc {
    slot_size: u64,
    slot_count: u32,
    _pad: u32,
    meta_offset: u64,
    data_offset: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RingDesc {
    pub(crate) stride: u32,
    pub(crate) capacity: u32,
    pub(crate) ctrl_offset: u64,
    pub(crate) data_offset: u64,
    pub(crate) data_len: u64,
}

#[repr(C)]
struct Superblock {
    magic: AtomicU64,
    version: u32,
    page_align: u32,
    total_len: u64,
    generation: u32,
    flags: u32,
    /// Counts publishes across the arena, so slots can be ordered oldest first.
    publish_counter: AtomicU64,
    class_count: u32,
    ring_count: u32,
    classes: [ClassDesc; MAX_CLASSES],
    rings: [RingDesc; MAX_RINGS],
    /// The reserved ring region, from the arena's start: both multiples of `PAGE_ALIGN`.
    ring_region_offset: u64,
    ring_region_len: u64,
    /// Entries reserved in `carved` so far (an entry is valid once its generation is nonzero).
    carved_count: AtomicU32,
    _carved_pad: u32,
    /// The rings carved from the ring region, in carving order (the order is the ring's index): append-only, each
    /// entry written in full and then published by a Release store of its generation, and never rewritten.
    carved: [CarvedEntry; MAX_CARVED],
}

/// The most rings an arena's ring region holds.
pub const MAX_CARVED: usize = 256;

/// One carved ring's record in the superblock.
#[repr(C)]
struct CarvedEntry {
    ctrl_offset: AtomicU64,
    data_offset: AtomicU64,
    stride: AtomicU32,
    capacity: AtomicU32,
    /// 0 while the entry is being written; the ring's generation once it's published.
    generation: AtomicU32,
    _pad: u32,
}

const _: () = assert!(size_of::<Superblock>() <= PAGE_ALIGN);

#[repr(C, align(64))]
struct SlotMeta {
    /// `generation << 32 | pins << 8 | renews << 2 | state`.
    word: AtomicU64,
    /// The newest pin's deadline, in monotonic nanoseconds.
    lease_expiry_ns: AtomicU64,
    /// Where the slot is, from the start of the arena (fixed at creation).
    offset: u64,
    /// How many bytes of the slot hold data.
    len: AtomicU64,
    /// Who claimed the slot: a stream, lease, or worker id.
    owner: AtomicU64,
    /// When it was published, from the superblock's publish counter.
    seq: AtomicU64,
    /// The generation the tier worker has a Tier 1 copy of (0 for none), so a late mark for an old generation can't
    /// make the slot's next use look flushed.
    flushed: AtomicU32,
    /// The generation a purge is taking back (0 for none). While it matches, no reclaim frees the slot: only the
    /// purge does, after zeroing it, so a purged payload's bytes are never left in a freed slot.
    purge: AtomicU32,
}

const _: () = assert!(size_of::<SlotMeta>() == 64);

/// A ring's control block: `head` and `started` alone on the first cache line (the writer stores them on every
/// record), and everything readers write (`waiters`) or rarely change on the second.
#[repr(C, align(128))]
pub(crate) struct RingCtrl {
    /// Records published so far; the next record goes at `head % capacity`.
    pub(crate) head: AtomicU64,
    /// Records the writer has started writing: `head`, or `head + 1` mid-write.
    pub(crate) started: AtomicU64,
    _hot_line: [u8; 48],
    /// The low 32 bits of `head`, stored after each record is published: the sequence word live readers block on
    /// (see [`crate::SeqWait`]).
    pub(crate) wait_word: AtomicU32,
    /// How many readers are blocked on `wait_word`, so the writer skips the wake syscall when it's 0.
    pub(crate) waiters: AtomicU32,
    /// The claim token of the [`crate::RingWriter`] holding the ring (unique per claim, never 0), or 0 when free.
    pub(crate) writer: AtomicU64,
    /// A carved ring's generation (0 for a ring in the superblock's table, and for a block not carved yet). Set when
    /// the ring is carved; readers and writers check it, so one built for an older carve of the block never reads or
    /// writes a newer one's records.
    pub(crate) generation: AtomicU32,
}

const _: () = assert!(size_of::<RingCtrl>() == 128 && std::mem::offset_of!(RingCtrl, wait_word) == 64);

const fn pack(generation: u32, pins: u32, state: u64) -> u64 {
    ((generation as u64) << 32) | ((pins as u64) << 8) | state
}

const fn generation_of(word: u64) -> u32 {
    (word >> 32) as u32
}

const fn pins_of(word: u64) -> u32 {
    ((word >> 8) & 0xff_ffff) as u32
}

const fn state_of(word: u64) -> u64 {
    word & 0x3
}

/// The word with its renew counter (the state byte's top 6 bits) bumped, so any compare-and-swap that read the old
/// word fails and rereads the lease expiry.
const fn touched(word: u64) -> u64 {
    let counter = ((word >> 2) + 1) & 0x3f;
    (word & !0xfc) | (counter << 2)
}

const fn next_generation(g: u32) -> u32 {
    match g.wrapping_add(1) {
        0 => 1,
        n => n,
    }
}

/// What happened when a Reclaiming slot was checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reclaimed {
    /// Freed; no reader held it.
    Freed,
    /// Freed although readers still held pins, because every pin's lease had expired (a slow consumer eviction).
    Evicted {
        /// How many pins were dropped.
        pins: u32,
    },
    /// Still pinned by a live lease (or waiting on a flush); it's freed when the last pin goes.
    Pinned,
}

/// A slot [`Arena::claim_overwriting`] reused before the tier worker flushed it: a gap in Tier 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Overwritten {
    /// The overwritten slot, as it was (its generation is now stale).
    pub handle: SlotHandle,
    /// Who had claimed it.
    pub owner: u64,
}

/// What [`Arena::purge`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Purged {
    /// Zeroed and freed.
    Zeroed,
    /// Taken out of use, but a live lease still holds it: call again once the lease is released or expires.
    Pending,
}

/// A slot's state, as [`Arena::scan`] sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotState {
    /// Free to claim.
    Free,
    /// Claimed and being written.
    Writing,
    /// Published: readers can lease it.
    Ready,
    /// On its way to free, waiting for its pins to go.
    Reclaiming,
}

/// One slot as [`Arena::scan`] saw it (a snapshot: it can change right after).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotScan {
    /// The slot, at the generation it had.
    pub handle: SlotHandle,
    /// Its state.
    pub state: SlotState,
    /// Pins held on it.
    pub pins: u32,
    /// When it was published (the arena's publish counter), for oldest first.
    pub seq: u64,
    /// Whether the tier worker has a Tier 1 copy of this generation.
    pub flushed: bool,
}

/// A shared memory arena. Cheap to clone; every clone shares one mapping, which is unmapped when the last clone,
/// [`WriteSlot`], [`Lease`], or ring handle is dropped.
#[derive(Clone)]
pub struct Arena {
    pub(crate) inner: Arc<Inner>,
}

pub(crate) struct Inner {
    map: Mapping,
    name: String,
    classes: Vec<ClassDesc>,
    pub(crate) rings: Vec<RingDesc>,
    /// The reserved ring region: (offset, length).
    pub(crate) ring_region: (u64, u64),
    generation: u32,
    requires_flush: bool,
    /// Where each class's next claim starts scanning, so slots are used in turn rather than the first one always.
    claim_hint: Vec<AtomicU32>,
}

impl Arena {
    /// Creates a new arena under `name` (at most 31 bytes, like `/cnt.hot.1`), sized in one go (macOS allows a
    /// single `ftruncate` per shared memory object). Fails if the name exists.
    pub fn create(name: &str, config: &ArenaConfig) -> Result<Self, Error> {
        let (total, classes, rings, region) = layout(config)?;
        let map = Mapping::create(name, total)?;
        let flags = if config.reclaim_requires_flush { FLAG_RECLAIM_REQUIRES_FLUSH } else { 0 };
        let sb = map.base().cast::<Superblock>();
        // SAFETY: the mapping is at least one page, page aligned, zeroed, and not yet visible to readers (they wait
        // for the magic, stored last). Plain writes initialize the non-atomic fields.
        unsafe {
            (&raw mut (*sb).version).write(VERSION);
            (&raw mut (*sb).page_align).write(PAGE_ALIGN as u32);
            (&raw mut (*sb).total_len).write(total as u64);
            (&raw mut (*sb).generation).write(config.generation);
            (&raw mut (*sb).flags).write(flags);
            (&raw mut (*sb).class_count).write(classes.len() as u32);
            (&raw mut (*sb).ring_count).write(rings.len() as u32);
            for (i, c) in classes.iter().enumerate() {
                (&raw mut (*sb).classes[i]).write(*c);
            }
            for (i, r) in rings.iter().enumerate() {
                (&raw mut (*sb).rings[i]).write(*r);
            }
            (&raw mut (*sb).ring_region_offset).write(region.0);
            (&raw mut (*sb).ring_region_len).write(region.1);
            for c in &classes {
                for index in 0..c.slot_count {
                    let meta = map.base().add(c.meta_offset as usize + index as usize * 64).cast::<SlotMeta>();
                    (&raw mut (*meta).offset).write(c.data_offset + u64::from(index) * c.slot_size);
                    (&raw mut (*meta).word).write(AtomicU64::new(pack(1, 0, FREE)));
                }
            }
        }
        let arena = Self::from_parts(map, name, classes, rings, region, config.generation, flags);
        arena.superblock().magic.store(MAGIC, Ordering::Release);
        Ok(arena)
    }

    /// Opens an existing arena. Returns [`Error::NotReady`] while its creator is still setting it up, so a reader
    /// that starts first can retry.
    pub fn open(name: &str) -> Result<Self, Error> {
        Self::from_mapping(Mapping::open(name)?, name)
    }

    /// Opens an arena from a file descriptor, the way a client maps the arenas the broker passes with `Welcome`.
    /// `name` is only the arena's label (from `Welcome`); the fd decides what's mapped. The mapping keeps its own
    /// reference, so `fd` may be closed afterward. Validates the superblock as [`Arena::open`] does.
    pub fn from_fd(fd: BorrowedFd<'_>, name: &str) -> Result<Self, Error> {
        // An fd handed over is never mid-setup (the broker sends finished arenas), so an empty or oddly sized object
        // isn't "not ready yet": it isn't an arena.
        let map = Mapping::from_fd(fd).map_err(|e| if matches!(e, Error::NotReady) { Error::BadArena } else { e })?;
        Self::from_mapping(map, name)
    }

    fn from_mapping(map: Mapping, name: &str) -> Result<Self, Error> {
        // SAFETY: the magic is the superblock's first field, an aligned AtomicU64 at the start of the mapping. The
        // rest of the superblock may still be being written until the magic is set, so nothing else is read first.
        let magic = unsafe { &*map.base().cast::<AtomicU64>() };
        match magic.load(Ordering::Acquire) {
            0 => return Err(Error::NotReady),
            MAGIC => {}
            _ => return Err(Error::BadArena),
        }
        // SAFETY: the creator stored the magic (Release) after writing every non-atomic field, and never writes them
        // again, so reading them after our Acquire load doesn't race.
        let sb = unsafe { &*map.base().cast::<Superblock>() };
        if sb.version != VERSION
            || sb.page_align as usize != PAGE_ALIGN
            || sb.total_len != map.size() as u64
            || sb.class_count as usize > MAX_CLASSES
            || sb.ring_count as usize > MAX_RINGS
        {
            return Err(Error::BadArena);
        }
        let classes = sb.classes[..sb.class_count as usize].to_vec();
        let rings = sb.rings[..sb.ring_count as usize].to_vec();
        let (generation, flags) = (sb.generation, sb.flags);
        let region = (sb.ring_region_offset, sb.ring_region_len);
        let len = map.size() as u64;
        let fits = |offset: u64, size: u128| u128::from(offset) + size <= u128::from(len);
        if !(is_aligned_16k(region.0 as usize) && is_aligned_16k(region.1 as usize) && fits(region.0, region.1.into()))
        {
            return Err(Error::BadArena);
        }
        let sane = classes.iter().all(|c| {
            c.slot_size > 0
                && is_aligned_16k(c.slot_size as usize)
                && is_aligned_16k(c.data_offset as usize)
                && c.meta_offset.is_multiple_of(64)
                && fits(c.meta_offset, u128::from(c.slot_count) * 64)
                && fits(c.data_offset, u128::from(c.slot_count) * u128::from(c.slot_size))
        }) && rings.iter().all(|r| {
            r.stride > 0
                && r.capacity > 0
                && is_aligned_16k(r.data_offset as usize)
                && u128::from(r.stride) * u128::from(r.capacity) <= u128::from(r.data_len)
                && r.ctrl_offset.is_multiple_of(128)
                && fits(r.ctrl_offset, size_of::<RingCtrl>() as u128)
                && fits(r.data_offset, u128::from(r.data_len))
        });
        if !sane {
            return Err(Error::BadArena);
        }
        Ok(Self::from_parts(map, name, classes, rings, region, generation, flags))
    }

    /// Removes an arena's name. Processes that have it open keep their mappings.
    pub fn unlink(name: &str) -> Result<(), Error> {
        sys::unlink(name)
    }

    fn from_parts(
        map: Mapping,
        name: &str,
        classes: Vec<ClassDesc>,
        rings: Vec<RingDesc>,
        ring_region: (u64, u64),
        generation: u32,
        flags: u32,
    ) -> Self {
        let claim_hint = classes.iter().map(|_| AtomicU32::new(0)).collect();
        let requires_flush = flags & FLAG_RECLAIM_REQUIRES_FLUSH != 0;
        let inner =
            Inner { map, name: name.to_owned(), classes, rings, ring_region, generation, requires_flush, claim_hint };
        Self { inner: Arc::new(inner) }
    }

    /// The arena's shared memory name.
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// The arena's generation (see [`ArenaConfig::generation`]).
    pub fn generation(&self) -> u32 {
        self.inner.generation
    }

    /// The arena's size in bytes.
    pub fn len(&self) -> usize {
        self.inner.map.size()
    }

    /// Always false: an arena has at least its superblock.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The mapping's base address in this process (16 KB aligned).
    pub fn base_ptr(&self) -> *const u8 {
        self.inner.map.base()
    }

    /// How many slab classes the arena has.
    pub fn class_count(&self) -> usize {
        self.inner.classes.len()
    }

    /// A class's slot size (a multiple of [`PAGE_ALIGN`]) and slot count.
    pub fn class(&self, class: u16) -> Result<SlabClass, Error> {
        let c = self.inner.classes.get(usize::from(class)).ok_or(Error::NoSuchSlot)?;
        Ok(SlabClass { slot_size: c.slot_size as usize, slot_count: c.slot_count })
    }

    /// How many rings the arena has.
    pub fn ring_count(&self) -> usize {
        self.inner.rings.len()
    }

    /// A slot's byte offset from the start of the arena (always a multiple of [`PAGE_ALIGN`]).
    pub fn slot_offset(&self, class: u16, index: u32) -> Result<usize, Error> {
        let c = self.inner.classes.get(usize::from(class)).ok_or(Error::NoSuchSlot)?;
        if index >= c.slot_count {
            return Err(Error::NoSuchSlot);
        }
        Ok(c.data_offset as usize + index as usize * c.slot_size as usize)
    }

    /// Claims a free slot in `class` for writing, tagged with `owner`. If none is free, first finishes reclaiming
    /// any Reclaiming slots whose pins are gone or expired. [`Error::Full`] if every slot is still in use.
    pub fn claim(&self, class: u16, owner: u64) -> Result<WriteSlot, Error> {
        let c = *self.inner.classes.get(usize::from(class)).ok_or(Error::NoSuchSlot)?;
        if let Some(slot) = self.try_claim(class, &c, owner) {
            return Ok(slot);
        }
        for index in 0..c.slot_count {
            let word = self.meta(usize::from(class), index).word.load(Ordering::SeqCst);
            if state_of(word) == RECLAIMING {
                self.try_free(class, index, generation_of(word));
            }
        }
        self.try_claim(class, &c, owner).ok_or(Error::Full)
    }

    /// Like [`Arena::claim`], but when nothing is free it reclaims the oldest Ready slot no live lease pins, flushed
    /// or not, so the producer never waits. For a capture stream's own slot ring. Also returns every slot it
    /// overwrote that wasn't flushed to Tier 1 (gaps in the archive): usually none or one, more only if another
    /// producer in the class took a slot this call freed. [`Error::Full`] only if every slot is being written or
    /// pinned.
    pub fn claim_overwriting(&self, class: u16, owner: u64) -> Result<(WriteSlot, Vec<Overwritten>), Error> {
        self.claim_reusing(class, owner, false)
    }

    /// Like [`Arena::claim`], but when nothing is free it reuses the oldest Ready slot no live lease pins that is
    /// already flushed to Tier 1, so a producer that must never overwrite unflushed data (an encoded camera, whose
    /// access units depend on each other) still recycles its ring without waiting for the broker's reclaim sweep.
    /// [`Error::Full`] when every slot is being written, pinned, or not yet flushed.
    pub fn claim_recycling(&self, class: u16, owner: u64) -> Result<WriteSlot, Error> {
        self.claim_reusing(class, owner, true).map(|(slot, _)| slot)
    }

    fn claim_reusing(
        &self,
        class: u16,
        owner: u64,
        flushed_only: bool,
    ) -> Result<(WriteSlot, Vec<Overwritten>), Error> {
        let c = *self.inner.classes.get(usize::from(class)).ok_or(Error::NoSuchSlot)?;
        let mut gaps = Vec::new();
        for _ in 0..=c.slot_count {
            match self.claim(class, owner) {
                Err(Error::Full) => {}
                other => return other.map(|slot| (slot, gaps)),
            }
            let now = sys::monotonic_ns();
            let oldest = (0..c.slot_count)
                .filter_map(|index| {
                    let meta = self.meta(usize::from(class), index);
                    let word = meta.word.load(Ordering::SeqCst);
                    let unpinned = pins_of(word) == 0 || meta.lease_expiry_ns.load(Ordering::SeqCst) <= now;
                    let flushed = !flushed_only || meta.flushed.load(Ordering::SeqCst) == generation_of(word);
                    (state_of(word) == READY && unpinned && flushed)
                        .then(|| (meta.seq.load(Ordering::SeqCst), index, word))
                })
                .min();
            let Some((_, index, word)) = oldest else { return Err(Error::Full) };
            let handle = SlotHandle { class, index, generation: generation_of(word) };
            let meta = self.meta(usize::from(class), index);
            let owner = meta.owner.load(Ordering::SeqCst);
            if !self.reclaim_if_unpinned(handle) {
                continue; // a reader or another producer got there first; look again
            }
            // Overwriting doesn't wait for a flush, whatever the arena's policy for evicting expired pins.
            let freed = match self.try_free_with(class, index, handle.generation, false) {
                Reclaimed::Freed | Reclaimed::Evicted { .. } => true,
                // A reader renewed between our reclaim and the free, so its pin is live again. Hand the slot back
                // to Ready rather than leave it Reclaiming, where only claim()'s sweep (under the arena's flush
                // policy) would ever free it. If the reader released meanwhile and that freed it, it's gone.
                Reclaimed::Pinned => !self.unreclaim(handle),
            };
            if freed && meta.flushed.load(Ordering::SeqCst) != handle.generation {
                gaps.push(Overwritten { handle, owner });
            }
        }
        Err(Error::Full)
    }

    /// Moves a slot this generation put in Reclaiming back to Ready, keeping its pins. Returns false if it's no longer
    /// Reclaiming in that generation (its last pin went and freed it).
    fn unreclaim(&self, handle: SlotHandle) -> bool {
        let meta = self.meta(usize::from(handle.class), handle.index);
        let mut word = meta.word.load(Ordering::SeqCst);
        loop {
            if generation_of(word) != handle.generation || state_of(word) != RECLAIMING {
                return false;
            }
            let ready = pack(handle.generation, pins_of(word), READY);
            match meta.word.compare_exchange(word, ready, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => return true,
                Err(w) => word = w,
            }
        }
    }

    /// Moves a Ready slot to Reclaiming, but only while no live lease pins it (no pins, or every pin expired).
    /// Returns whether it did.
    fn reclaim_if_unpinned(&self, handle: SlotHandle) -> bool {
        let meta = self.meta(usize::from(handle.class), handle.index);
        let mut word = meta.word.load(Ordering::SeqCst);
        loop {
            if generation_of(word) != handle.generation || state_of(word) != READY {
                return false;
            }
            // Read after the word: a lease raises the expiry before it pins, and pinning changes the word, so either
            // the CAS below fails or this read sees the new deadline.
            let expired = meta.lease_expiry_ns.load(Ordering::SeqCst) <= sys::monotonic_ns();
            if pins_of(word) > 0 && !expired {
                return false;
            }
            let next = pack(handle.generation, pins_of(word), RECLAIMING);
            match meta.word.compare_exchange(word, next, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => return true,
                Err(w) => word = w,
            }
        }
    }

    /// Starts reclaiming a published slot: new leases are refused, and it's freed when its last pin goes (or when
    /// every pin's lease has expired). Returns what happened right away.
    pub fn reclaim(&self, handle: SlotHandle) -> Result<Reclaimed, Error> {
        self.reclaim_with(handle, self.inner.requires_flush)
    }

    fn reclaim_with(&self, handle: SlotHandle, requires_flush: bool) -> Result<Reclaimed, Error> {
        let meta = self.meta_for(handle)?;
        let mut word = meta.word.load(Ordering::SeqCst);
        loop {
            if generation_of(word) != handle.generation || state_of(word) != READY {
                return Err(Error::Stale);
            }
            let next = pack(handle.generation, pins_of(word), RECLAIMING);
            match meta.word.compare_exchange(word, next, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => break,
                Err(w) => word = w,
            }
        }
        Ok(self.try_free_with(handle.class, handle.index, handle.generation, requires_flush))
    }

    /// Every slot in `class`, as it is now: for a policy that reclaims the oldest published slots when free ones run
    /// low, and that notices slots that moved on (reclaimed, overwritten) since it last looked.
    ///
    /// # Errors
    /// [`Error::NoSuchSlot`] if the class doesn't exist.
    pub fn scan(&self, class: u16) -> Result<Vec<SlotScan>, Error> {
        let c = *self.inner.classes.get(usize::from(class)).ok_or(Error::NoSuchSlot)?;
        Ok((0..c.slot_count)
            .map(|index| {
                let meta = self.meta(usize::from(class), index);
                let word = meta.word.load(Ordering::SeqCst);
                let generation = generation_of(word);
                let state = match state_of(word) {
                    FREE => SlotState::Free,
                    WRITING => SlotState::Writing,
                    READY => SlotState::Ready,
                    _ => SlotState::Reclaiming,
                };
                SlotScan {
                    handle: SlotHandle { class, index, generation },
                    state,
                    pins: pins_of(word),
                    seq: meta.seq.load(Ordering::SeqCst),
                    flushed: meta.flushed.load(Ordering::SeqCst) == generation,
                }
            })
            .collect())
    }

    /// Takes a published (or reclaiming) slot back for a purge, whose payload must not stay readable: it stops being
    /// leasable at once (Reclaiming, marked as purging so no reclaim frees it unzeroed), and its bytes are zeroed and
    /// the slot freed once no live lease holds it. A lease promises its bytes don't change while it's live, and
    /// readers rely on that in place (numpy and DLPack views, GPU mappings), so a slot still pinned is left for a
    /// later call: [`Purged::Pending`]. An expired lease no longer holds it.
    ///
    /// # Errors
    /// [`Error::Stale`] if the slot already moved on from `handle`'s generation (its bytes are someone else's now).
    pub fn purge(&self, handle: SlotHandle) -> Result<Purged, Error> {
        let meta = self.meta_for(handle)?;
        // Marked first, so a reclaim that sees the slot Reclaiming from here on leaves it to us. A stale mark (an old
        // generation) never matches the slot's next use.
        meta.purge.store(handle.generation, Ordering::SeqCst);
        let mut word = meta.word.load(Ordering::SeqCst);
        loop {
            if generation_of(word) != handle.generation {
                return Err(Error::Stale);
            }
            match state_of(word) {
                RECLAIMING => break,
                READY => {
                    let next = pack(handle.generation, pins_of(word), RECLAIMING);
                    match meta.word.compare_exchange(word, next, Ordering::SeqCst, Ordering::SeqCst) {
                        Ok(_) => break,
                        Err(w) => word = w,
                    }
                }
                _ => return Err(Error::Stale),
            }
        }
        // Zero only once no live lease holds it: take it to the next generation in Writing, so no producer claims it
        // while it's zeroed and every old handle is stale.
        let next = next_generation(handle.generation);
        let mut word = meta.word.load(Ordering::SeqCst);
        loop {
            if generation_of(word) != handle.generation || state_of(word) != RECLAIMING {
                return Err(Error::Stale);
            }
            let expired = meta.lease_expiry_ns.load(Ordering::SeqCst) <= sys::monotonic_ns();
            if pins_of(word) > 0 && !expired {
                return Ok(Purged::Pending);
            }
            match meta.word.compare_exchange(word, pack(next, 0, WRITING), Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => break,
                Err(w) => word = w,
            }
        }
        // The zero stores must not move ahead of the state change on weakly ordered CPUs (AArch64).
        fence(Ordering::SeqCst);
        let offset = self.slot_offset(handle.class, handle.index)?;
        let len = self.inner.classes[usize::from(handle.class)].slot_size as usize;
        // SAFETY: the slot lies inside the mapping (slot_offset checked it) and is a whole number of 16 KB pages, so
        // 8-byte words cover it exactly. It's in Writing under a generation only this call holds, so no producer
        // writes it, and no live lease holds it (checked above), so no reader may be reading it in place. A reader
        // whose lease expired isn't promised the bytes any more, and checks `Lease::is_valid` after reading; the
        // stores are atomic so even such a late read is defined.
        unsafe {
            let base = self.inner.map.base().add(offset).cast::<AtomicU64>();
            for i in 0..len / 8 {
                (*base.add(i)).store(0, Ordering::Relaxed);
            }
        }
        meta.len.store(0, Ordering::SeqCst);
        meta.purge.store(0, Ordering::SeqCst);
        // Release: a producer that claims the slot sees the zeroes.
        meta.word.store(pack(next, 0, FREE), Ordering::SeqCst);
        Ok(Purged::Zeroed)
    }

    /// Records that the tier worker has a Tier 1 copy of the slot, so an expired pin may be evicted and an overwrite
    /// isn't a gap. [`Error::Stale`] if the slot was reused since `handle` was made (the mark then has no effect).
    pub fn mark_flushed(&self, handle: SlotHandle) -> Result<(), Error> {
        let meta = self.meta_for(handle)?;
        // The generation, not a flag: a late mark for an old generation never matches the slot's next use.
        meta.flushed.store(handle.generation, Ordering::SeqCst);
        if generation_of(meta.word.load(Ordering::SeqCst)) != handle.generation {
            return Err(Error::Stale);
        }
        Ok(())
    }

    /// Pins a published slot for reading until `ttl` from now (at most [`MAX_LEASE`]; standing readers renew).
    /// [`Error::Stale`] if the slot is being reclaimed or was reused since `handle` was made.
    pub fn lease(&self, handle: SlotHandle, ttl: Duration) -> Result<Lease, Error> {
        let ttl_ns = lease_ns(ttl)?;
        let meta = self.meta_for(handle)?;
        let mut word = meta.word.load(Ordering::SeqCst);
        if generation_of(word) != handle.generation || state_of(word) != READY {
            return Err(Error::Stale);
        }
        // Raise the expiry before the pin exists, so a reclaimer that sees the pin also sees a live deadline.
        let deadline = sys::monotonic_ns() + ttl_ns;
        meta.lease_expiry_ns.fetch_max(deadline, Ordering::SeqCst);
        loop {
            if generation_of(word) != handle.generation || state_of(word) != READY {
                return Err(Error::Stale);
            }
            let pins = pins_of(word);
            if pins == MAX_PINS {
                return Err(Error::Busy);
            }
            let next = pack(handle.generation, pins + 1, READY);
            match meta.word.compare_exchange(word, next, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => break,
                Err(w) => word = w,
            }
        }
        Ok(Lease {
            arena: self.clone(),
            handle,
            deadline_ns: AtomicU64::new(deadline),
            offset: self.slot_offset(handle.class, handle.index)?,
            len: meta.len.load(Ordering::SeqCst) as usize,
            released: AtomicBool::new(false),
        })
    }

    /// How many of a class's slots can't be taken back without losing something: being written or reclaimed, or Ready
    /// with a live pin (a reader holds it) or with no Tier 1 copy yet (the tier worker hasn't flushed it). A capture
    /// class runs full by design, since its producer overwrites the oldest Ready slot; what matters for its health is
    /// this count, not [`Arena::used`]. A snapshot; slots change state as it counts.
    pub fn held(&self, class: u16) -> Result<u32, Error> {
        let c = self.inner.classes.get(usize::from(class)).ok_or(Error::NoSuchSlot)?;
        let now = sys::monotonic_ns();
        Ok((0..c.slot_count)
            .filter(|&i| {
                let meta = self.meta(usize::from(class), i);
                let word = meta.word.load(Ordering::SeqCst);
                match state_of(word) {
                    FREE => false,
                    READY => {
                        let pinned = pins_of(word) > 0 && meta.lease_expiry_ns.load(Ordering::SeqCst) > now;
                        pinned || meta.flushed.load(Ordering::SeqCst) != generation_of(word)
                    }
                    _ => true, // being written or reclaimed
                }
            })
            .count() as u32)
    }

    /// How many of a class's slots are in use (being written, ready, or being reclaimed): what a status check reports
    /// against [`SlabClass::slot_count`]. A snapshot; slots change state as it counts.
    pub fn used(&self, class: u16) -> Result<u32, Error> {
        let c = self.inner.classes.get(usize::from(class)).ok_or(Error::NoSuchSlot)?;
        Ok((0..c.slot_count)
            .filter(|&i| state_of(self.meta(usize::from(class), i).word.load(Ordering::SeqCst)) != FREE)
            .count() as u32)
    }

    /// How many pins the slot holds, live or expired (0 if `handle` is stale).
    pub fn pin_count(&self, handle: SlotHandle) -> Result<u32, Error> {
        let word = self.meta_for(handle)?.word.load(Ordering::SeqCst);
        Ok(if generation_of(word) == handle.generation { pins_of(word) } else { 0 })
    }

    /// The owner tag the slot was claimed with. [`Error::Stale`] if the slot was reused since `handle` was made.
    pub fn owner(&self, handle: SlotHandle) -> Result<u64, Error> {
        let meta = self.meta_for(handle)?;
        let owner = meta.owner.load(Ordering::SeqCst);
        let word = meta.word.load(Ordering::SeqCst);
        if generation_of(word) != handle.generation || state_of(word) == FREE {
            return Err(Error::Stale);
        }
        Ok(owner)
    }

    fn try_claim(&self, class: u16, c: &ClassDesc, owner: u64) -> Option<WriteSlot> {
        let hint = &self.inner.claim_hint[usize::from(class)];
        let start = hint.load(Ordering::Relaxed);
        for i in 0..c.slot_count {
            let index = (start + i) % c.slot_count;
            let meta = self.meta(usize::from(class), index);
            let word = meta.word.load(Ordering::SeqCst);
            if state_of(word) != FREE {
                continue;
            }
            let generation = generation_of(word);
            let writing = pack(generation, 0, WRITING);
            if meta.word.compare_exchange(word, writing, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                meta.owner.store(owner, Ordering::SeqCst);
                meta.flushed.store(0, Ordering::SeqCst);
                meta.lease_expiry_ns.store(0, Ordering::SeqCst);
                meta.len.store(0, Ordering::SeqCst);
                hint.store((index + 1) % c.slot_count, Ordering::Relaxed);
                return Some(WriteSlot {
                    arena: self.clone(),
                    handle: SlotHandle { class, index, generation },
                    done: false,
                });
            }
        }
        None
    }

    /// Frees a Reclaiming slot if no pin holds it, or if every pin has expired (and, when the arena requires it,
    /// the slot is flushed).
    fn try_free(&self, class: u16, index: u32, generation: u32) -> Reclaimed {
        self.try_free_with(class, index, generation, self.inner.requires_flush)
    }

    fn try_free_with(&self, class: u16, index: u32, generation: u32, requires_flush: bool) -> Reclaimed {
        let meta = self.meta(usize::from(class), index);
        // A slot being purged is freed only by the purge, once it's zeroed.
        if meta.purge.load(Ordering::SeqCst) == generation {
            return Reclaimed::Pinned;
        }
        let mut word = meta.word.load(Ordering::SeqCst);
        loop {
            if generation_of(word) != generation || state_of(word) != RECLAIMING {
                return Reclaimed::Pinned; // someone else freed it, or it isn't ours to free
            }
            let pins = pins_of(word);
            if pins > 0 {
                let expired = meta.lease_expiry_ns.load(Ordering::SeqCst) <= sys::monotonic_ns();
                let flushed = !requires_flush || meta.flushed.load(Ordering::SeqCst) == generation;
                if !(expired && flushed) {
                    return Reclaimed::Pinned;
                }
            }
            let free = pack(next_generation(generation), 0, FREE);
            match meta.word.compare_exchange(word, free, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) if pins == 0 => return Reclaimed::Freed,
                Ok(_) => return Reclaimed::Evicted { pins },
                Err(w) => word = w,
            }
        }
    }

    /// The rings carved so far, by index, as the superblock records them (only published entries).
    pub fn carved_rings(&self) -> Vec<crate::CarvedRing> {
        let sb = self.superblock();
        let count = (sb.carved_count.load(Ordering::Acquire) as usize).min(MAX_CARVED);
        sb.carved[..count]
            .iter()
            .filter_map(|e| {
                // Acquire: a published entry's other fields were written before its generation.
                let generation = e.generation.load(Ordering::Acquire);
                (generation != 0).then(|| crate::CarvedRing {
                    ctrl_offset: e.ctrl_offset.load(Ordering::Relaxed),
                    data_offset: e.data_offset.load(Ordering::Relaxed),
                    stride: e.stride.load(Ordering::Relaxed),
                    capacity: e.capacity.load(Ordering::Relaxed),
                    generation,
                })
            })
            .collect()
    }

    /// Records a carved ring in the superblock's table: reserves the next entry, writes it, then publishes it by
    /// storing its generation. Returns its index.
    pub(crate) fn append_carved(&self, ring: &crate::CarvedRing) -> Result<u16, Error> {
        let sb = self.superblock();
        let index = sb.carved_count.fetch_add(1, Ordering::SeqCst) as usize;
        if index >= MAX_CARVED {
            sb.carved_count.fetch_sub(1, Ordering::SeqCst);
            return Err(Error::BadConfig(format!("at most {MAX_CARVED} carved rings")));
        }
        // The entry was just reserved for this call alone; readers read its fields only after its generation (stored
        // last, with Release) is nonzero. Atomics, so no process ever reads a plain field another is writing.
        let e = &sb.carved[index];
        e.ctrl_offset.store(ring.ctrl_offset, Ordering::Relaxed);
        e.data_offset.store(ring.data_offset, Ordering::Relaxed);
        e.stride.store(ring.stride, Ordering::Relaxed);
        e.capacity.store(ring.capacity, Ordering::Relaxed);
        e.generation.store(ring.generation, Ordering::Release);
        u16::try_from(index).map_err(|_| Error::BadConfig("too many carved rings".into()))
    }

    fn superblock(&self) -> &Superblock {
        // SAFETY: the mapping starts with the superblock, and every field touched after setup is immutable or
        // atomic.
        unsafe { &*self.inner.map.base().cast::<Superblock>() }
    }

    fn meta(&self, class: usize, index: u32) -> &SlotMeta {
        let c = &self.inner.classes[class];
        debug_assert!(index < c.slot_count);
        // SAFETY: create and open checked that the class's directory lies inside the mapping and is 64-byte
        // aligned. SlotMeta's mutable fields are atomics; `offset` is written once, before the magic is published.
        unsafe { &*self.inner.map.base().add(c.meta_offset as usize + index as usize * 64).cast::<SlotMeta>() }
    }

    fn meta_for(&self, handle: SlotHandle) -> Result<&SlotMeta, Error> {
        let c = self.inner.classes.get(usize::from(handle.class)).ok_or(Error::NoSuchSlot)?;
        if handle.index >= c.slot_count {
            return Err(Error::NoSuchSlot);
        }
        Ok(self.meta(usize::from(handle.class), handle.index))
    }

    pub(crate) fn ring_ctrl(&self, desc: &RingDesc) -> &RingCtrl {
        // SAFETY: create and open checked the control block lies inside the mapping and is 128-byte aligned; it's
        // all atomics.
        unsafe { &*self.inner.map.base().add(desc.ctrl_offset as usize).cast::<RingCtrl>() }
    }

    pub(crate) fn ptr_at(&self, offset: usize) -> *mut u8 {
        debug_assert!(offset <= self.inner.map.size());
        // SAFETY: callers pass offsets from checked descriptors, inside the mapping.
        unsafe { self.inner.map.base().add(offset) }
    }
}

/// The longest lease. Standing readers renew well before it runs out.
///
/// Kept short because the slot's expiry is the newest deadline of any lease taken on it, released or not: a long
/// lease, even released, delays evicting a stalled reader by up to its ttl. Standing readers renew instead.
pub const MAX_LEASE: Duration = Duration::from_secs(60);

fn lease_ns(ttl: Duration) -> Result<u64, Error> {
    if ttl.is_zero() || ttl > MAX_LEASE {
        return Err(Error::BadLease(ttl));
    }
    Ok(ttl.as_nanos() as u64)
}

/// Lays out an arena: returns its total size and the class and ring descriptors.
type Layout = (usize, Vec<ClassDesc>, Vec<RingDesc>, (u64, u64));

fn layout(config: &ArenaConfig) -> Result<Layout, Error> {
    if config.classes.len() > MAX_CLASSES {
        return Err(Error::BadConfig(format!("at most {MAX_CLASSES} slab classes")));
    }
    if config.rings.len() > MAX_RINGS {
        return Err(Error::BadConfig(format!("at most {MAX_RINGS} rings")));
    }
    let overflow = || Error::TooLarge;
    let add = |a: usize, b: usize| a.checked_add(b).ok_or_else(overflow);
    let mul = |a: usize, b: usize| a.checked_mul(b).ok_or_else(overflow);

    let mut offset = checked_align_up_16k(size_of::<Superblock>()).ok_or_else(overflow)?;
    let mut classes = Vec::with_capacity(config.classes.len());
    for c in &config.classes {
        if c.slot_size == 0 || c.slot_count == 0 {
            return Err(Error::BadConfig("a slab class needs a nonzero slot size and count".into()));
        }
        let slot_size = checked_align_up_16k(c.slot_size).ok_or_else(overflow)?;
        classes.push(ClassDesc {
            slot_size: slot_size as u64,
            slot_count: c.slot_count,
            meta_offset: offset as u64,
            ..ClassDesc::default()
        });
        offset = add(offset, mul(c.slot_count as usize, size_of::<SlotMeta>())?)?;
    }
    let mut rings = Vec::with_capacity(config.rings.len());
    offset = offset.next_multiple_of(size_of::<RingCtrl>());
    for r in &config.rings {
        if r.stride == 0 || r.capacity == 0 {
            return Err(Error::BadConfig("a ring needs a nonzero stride and capacity".into()));
        }
        let data_len = checked_align_up_16k(mul(r.stride as usize, r.capacity as usize)?).ok_or_else(overflow)?;
        rings.push(RingDesc {
            stride: r.stride,
            capacity: r.capacity,
            ctrl_offset: offset as u64,
            data_len: data_len as u64,
            ..RingDesc::default()
        });
        offset = add(offset, size_of::<RingCtrl>())?;
    }
    offset = checked_align_up_16k(offset).ok_or_else(overflow)?;
    for c in &mut classes {
        c.data_offset = offset as u64;
        offset = add(offset, mul(c.slot_size as usize, c.slot_count as usize)?)?;
    }
    for r in &mut rings {
        r.data_offset = offset as u64;
        offset = add(offset, r.data_len as usize)?;
    }
    let region_len = checked_align_up_16k(config.ring_region_bytes).ok_or_else(overflow)?;
    let region = (offset as u64, region_len as u64);
    offset = add(offset, region_len)?;
    debug_assert!(is_aligned_16k(offset));
    Ok((offset, classes, rings, region))
}

/// A claimed slot being written. [`WriteSlot::publish`] makes it readable; dropping it unpublished frees it.
pub struct WriteSlot {
    arena: Arena,
    handle: SlotHandle,
    done: bool,
}

impl WriteSlot {
    /// The slot's handle (valid for leases once published).
    pub fn handle(&self) -> SlotHandle {
        self.handle
    }

    /// The slot's byte offset in the arena (a multiple of [`PAGE_ALIGN`]).
    pub fn offset(&self) -> usize {
        self.arena.slot_offset(self.handle.class, self.handle.index).expect("a claimed slot exists")
    }

    /// The slot's size in bytes (a multiple of [`PAGE_ALIGN`]).
    pub fn capacity(&self) -> usize {
        self.arena.inner.classes[usize::from(self.handle.class)].slot_size as usize
    }

    /// The slot's bytes, to fill before publishing.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        let ptr = self.arena.ptr_at(self.offset());
        // SAFETY: the slot lies inside the mapping, and while it's in Writing nobody else reads or writes it (leases
        // need Ready, and only this WriteSlot moves it out of Writing).
        unsafe { std::slice::from_raw_parts_mut(ptr, self.capacity()) }
    }

    /// Publishes the first `len` bytes: readers can lease the slot from now on. Returns its handle, or
    /// [`Error::Stale`] if the slot was taken away meanwhile (the broker recovering a producer it thought crashed).
    pub fn publish(mut self, len: usize) -> Result<SlotHandle, Error> {
        if len > self.capacity() {
            return Err(Error::TooLong { len, capacity: self.capacity() });
        }
        let meta = self.arena.meta(usize::from(self.handle.class), self.handle.index);
        meta.len.store(len as u64, Ordering::SeqCst);
        let seq = self.arena.superblock().publish_counter.fetch_add(1, Ordering::SeqCst);
        meta.seq.store(seq, Ordering::SeqCst);
        let g = self.handle.generation;
        self.done = true; // published or taken away, there's nothing for Drop to undo
        meta.word
            .compare_exchange(pack(g, 0, WRITING), pack(g, 0, READY), Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|_| Error::Stale)?;
        Ok(self.handle)
    }
}

impl Drop for WriteSlot {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        let meta = self.arena.meta(usize::from(self.handle.class), self.handle.index);
        let g = self.handle.generation;
        let free = pack(next_generation(g), 0, FREE);
        meta.word.compare_exchange(pack(g, 0, WRITING), free, Ordering::SeqCst, Ordering::SeqCst).ok();
    }
}

/// A reader's pin on a published slot, valid until it's released or its deadline passes. Dropping it releases it.
///
/// While the lease is live the slot's bytes don't change. Once it expires, a producer may reuse the slot, so a
/// reader that may have stalled checks [`Lease::is_valid`] after reading.
pub struct Lease {
    arena: Arena,
    handle: SlotHandle,
    deadline_ns: AtomicU64,
    offset: usize,
    len: usize,
    released: AtomicBool,
}

impl Lease {
    /// The leased slot's handle.
    pub fn handle(&self) -> SlotHandle {
        self.handle
    }

    /// The slot's byte offset in the arena (a multiple of [`PAGE_ALIGN`]).
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// The published length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the published payload is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The first published byte, in this process (16 KB aligned).
    pub fn as_ptr(&self) -> *const u8 {
        self.arena.ptr_at(self.offset)
    }

    /// The arena this lease is in.
    pub fn arena(&self) -> &Arena {
        &self.arena
    }

    /// The published bytes, without copying.
    ///
    /// # Safety
    ///
    /// The lease must not expire while the slice is in use: once it does, a producer may overwrite the bytes. Renew
    /// it if the reader might take longer than its ttl.
    pub unsafe fn as_slice(&self) -> &[u8] {
        // SAFETY: the slot lies inside the mapping, and the caller promises the lease stays live, so nothing writes
        // these bytes meanwhile.
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len) }
    }

    /// Copies the published bytes into `out` (exactly `len()` bytes), then checks the lease was still valid, so
    /// the copy can't be torn. [`Error::Expired`] if it wasn't.
    pub fn copy_to(&self, out: &mut [u8]) -> Result<(), Error> {
        if out.len() != self.len {
            return Err(Error::WrongLength { expected: self.len, got: out.len() });
        }
        // SAFETY: source and destination are valid for len bytes and don't overlap (out is process memory).
        unsafe { ptr::copy_nonoverlapping(self.as_ptr(), out.as_mut_ptr(), self.len) };
        std::sync::atomic::fence(Ordering::SeqCst);
        if self.is_valid() { Ok(()) } else { Err(Error::Expired) }
    }

    /// Whether the lease still holds the slot: not released, not past its deadline, and the slot not reused.
    pub fn is_valid(&self) -> bool {
        if self.released.load(Ordering::SeqCst) {
            return false;
        }
        let word = self.arena.meta(usize::from(self.handle.class), self.handle.index).word.load(Ordering::SeqCst);
        generation_of(word) == self.handle.generation
            && pins_of(word) > 0
            && sys::monotonic_ns() < self.deadline_ns.load(Ordering::SeqCst)
    }

    /// Pushes the deadline out to `ttl` from now. [`Error::Expired`] if the lease already lapsed, was released, or
    /// lost its slot.
    pub fn renew(&self, ttl: Duration) -> Result<(), Error> {
        let ttl_ns = lease_ns(ttl)?;
        if !self.is_valid() {
            return Err(Error::Expired);
        }
        let meta = self.arena.meta(usize::from(self.handle.class), self.handle.index);
        let deadline = sys::monotonic_ns() + ttl_ns;
        meta.lease_expiry_ns.fetch_max(deadline, Ordering::SeqCst);
        // Bump the word's renew counter: an eviction that read the old expiry then fails its compare-and-swap and
        // rereads it, so a renew that returns Ok can't be evicted on the old deadline.
        let mut word = meta.word.load(Ordering::SeqCst);
        loop {
            let state = state_of(word);
            if generation_of(word) != self.handle.generation
                || pins_of(word) == 0
                || !(state == READY || state == RECLAIMING)
            {
                return Err(Error::Expired);
            }
            match meta.word.compare_exchange(word, touched(word), Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => break,
                Err(w) => word = w,
            }
        }
        self.deadline_ns.store(deadline, Ordering::SeqCst);
        Ok(())
    }

    /// Unpins the slot, freeing it if it's being reclaimed and this was the last pin. Returns whether the lease
    /// still held the slot (false if it was evicted or already released). Releasing twice is a no-op.
    pub fn release(&self) -> bool {
        if self.released.swap(true, Ordering::SeqCst) {
            return false;
        }
        let meta = self.arena.meta(usize::from(self.handle.class), self.handle.index);
        let g = self.handle.generation;
        let mut word = meta.word.load(Ordering::SeqCst);
        loop {
            let state = state_of(word);
            if generation_of(word) != g || pins_of(word) == 0 || !(state == READY || state == RECLAIMING) {
                return false; // evicted: the slot moved on without us
            }
            let pins = pins_of(word) - 1;
            // The last pin on a reclaiming slot frees it, unless a purge is taking it back: the purge zeroes it first.
            let purging = meta.purge.load(Ordering::SeqCst) == g;
            let next = if state == RECLAIMING && pins == 0 && !purging {
                pack(next_generation(g), 0, FREE)
            } else {
                pack(g, pins, state)
            };
            match meta.word.compare_exchange(word, next, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => return true,
                Err(w) => word = w,
            }
        }
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_state_word_round_trips() {
        let w = pack(u32::MAX, MAX_PINS, RECLAIMING);
        assert_eq!((generation_of(w), pins_of(w), state_of(w)), (u32::MAX, MAX_PINS, RECLAIMING));
    }

    #[test]
    fn a_renew_changes_the_word_but_nothing_else_in_it() {
        let w = pack(7, 3, READY);
        let once = touched(w);
        assert_ne!(once, w);
        assert_eq!((generation_of(once), pins_of(once), state_of(once)), (7, 3, READY));
        // The counter wraps within its 6 bits without spilling into the pin count.
        let wrapped = (0..64).fold(w, |word, _| touched(word));
        assert_eq!(wrapped, w);
    }

    #[test]
    fn generations_skip_zero() {
        assert_eq!(next_generation(1), 2);
        assert_eq!(next_generation(u32::MAX), 1);
    }

    #[test]
    fn layout_puts_every_slot_and_ring_on_a_16k_boundary() {
        let config = ArenaConfig {
            classes: vec![
                SlabClass { slot_size: 1, slot_count: 3 },
                SlabClass { slot_size: 3840 * 2160 * 3, slot_count: 5 },
                SlabClass { slot_size: PAGE_ALIGN + 1, slot_count: 7 },
            ],
            rings: vec![RingSpec { stride: 7, capacity: 1000 }, RingSpec { stride: 4096, capacity: 3 }],
            ..ArenaConfig::default()
        };
        let (total, classes, rings, _) = layout(&config).unwrap();
        assert!(is_aligned_16k(total));
        for c in &classes {
            assert!(is_aligned_16k(c.slot_size as usize));
            for i in 0..u64::from(c.slot_count) {
                assert!(is_aligned_16k((c.data_offset + i * c.slot_size) as usize));
            }
        }
        for r in &rings {
            assert!(is_aligned_16k(r.data_offset as usize));
            assert!(is_aligned_16k(r.data_len as usize));
            assert!(r.data_len >= u64::from(r.stride) * u64::from(r.capacity));
        }
        let last = rings.last().unwrap();
        assert_eq!(total as u64, last.data_offset + last.data_len);
    }

    #[test]
    fn layout_rejects_empty_and_oversized_configs() {
        let zero = ArenaConfig { classes: vec![SlabClass { slot_size: 0, slot_count: 1 }], ..ArenaConfig::default() };
        assert!(matches!(layout(&zero), Err(Error::BadConfig(_))));
        let huge = ArenaConfig {
            classes: vec![SlabClass { slot_size: usize::MAX / 2, slot_count: 4 }],
            ..ArenaConfig::default()
        };
        assert!(matches!(layout(&huge), Err(Error::TooLarge)));
    }
}
