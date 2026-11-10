//! Lock-free rings of fixed-size records, for streams like PCM audio, IMU samples, or slot handles.
//!
//! One writer and any number of readers. The writer never waits: when the ring is full it overwrites the oldest
//! record, and a reader that falls behind skips ahead and is told how many records it missed. Each reader keeps its
//! own cursor, in its own process.
//!
//! The ring's control block holds `head`, the number of records published, and `started`, the number the writer
//! has begun. Record `n` lives at `n % capacity`. The writer bumps `started`, fences, writes record `head`, then
//! publishes `head + 1`. A reader copies a record and then rereads `started` (a seqlock): if the writer had begun the
//! record that reuses the slot, the copy is discarded and the reader skips ahead. The design doc's "each record
//! carries a sequence number" is these two counters: a reader always knows a record's sequence number and sees
//! every gap. So readers always copy records out; rings carry small records, and large
//! payloads go in slab slots whose handles go through a ring.
//!
//! Record bytes are written and read with relaxed atomic accesses, never plain ones, so a reader copying a record
//! the writer is overwriting is a race the seqlock discards, not undefined behavior. Both sides use 8-byte words
//! when the record's address and length allow it, bytes otherwise; the choice depends only on the record's position
//! and the ring's stride, so the writer and every reader always use the same width.

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering, fence};
use std::time::{Duration, Instant};

use crate::arena::{Arena, RingCtrl, RingDesc};
use crate::{Error, PAGE_ALIGN, SeqWait};

/// A claim token unique to this claim: the process id in the high half, a per-process count in the low half, never
/// 0 (0 means the ring is free).
fn claim_token() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    static SALT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    // One clock sample per process, so a process that reuses a dead one's pid doesn't repeat its tokens, while the
    // count keeps this process's tokens distinct from each other (an odd multiplier is a bijection on 32 bits).
    let salt = *SALT.get_or_init(|| {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.subsec_nanos() | 1)
    });
    let n = NEXT.fetch_add(1, Ordering::Relaxed) as u32;
    let mut low = n.wrapping_mul(0x9E37_79B9) ^ salt;
    if low == 0 {
        low = n.wrapping_mul(0x9E37_79B9) ^ salt.rotate_left(16);
    }
    (u64::from(std::process::id()) << 32) | u64::from(low)
}

/// A ring carved from an arena's ring region while the arena is live (see [`Arena::carve_ring`]), as the broker
/// announces it. Readers and writers in any process build from these numbers, which are checked against the region
/// first; the superblock's own ring table never changes after the arena is made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CarvedRing {
    /// The control block, from the arena's start: the first page of the ring's span.
    pub ctrl_offset: u64,
    /// The records, from the arena's start: the page after the control block.
    pub data_offset: u64,
    /// Bytes per record.
    pub stride: u32,
    /// How many records the ring keeps before overwriting the oldest.
    pub capacity: u32,
    /// The carve's generation, checked against the control block's.
    pub generation: u32,
}

impl CarvedRing {
    /// The bytes its records take, rounded up to whole pages.
    fn data_len(&self) -> u64 {
        (u64::from(self.stride) * u64::from(self.capacity)).next_multiple_of(PAGE_ALIGN as u64)
    }

    /// The bytes of the ring region a ring of this shape takes: a page for its control block, then its records
    /// rounded up to whole pages. `None` if that overflows.
    #[must_use]
    pub fn span(stride: u32, capacity: u32) -> Option<u64> {
        let data = u64::from(stride).checked_mul(u64::from(capacity))?.checked_next_multiple_of(PAGE_ALIGN as u64)?;
        data.checked_add(PAGE_ALIGN as u64)
    }
}

/// Where a new reader starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadFrom {
    /// The oldest record still in the ring.
    Oldest,
    /// The next record written after the reader is made.
    Latest,
}

/// A record a reader got.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Record {
    /// The record's sequence number (0 for the first record ever written).
    pub seq: u64,
    /// Records skipped since the reader's previous read, because the writer overwrote them first.
    pub lost: u64,
}

impl Arena {
    /// Takes the ring's single writer. [`Error::RingBusy`] if another writer (in any process) holds it.
    pub fn ring_writer(&self, ring: usize) -> Result<RingWriter, Error> {
        let desc = *self.inner.rings.get(ring).ok_or(Error::NoSuchRing)?;
        let token = claim_token();
        self.ring_ctrl(&desc)
            .writer
            .compare_exchange(0, token, Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|_| Error::RingBusy(ring))?;
        Ok(RingWriter { arena: self.clone(), desc, generation: None, token })
    }

    /// Makes a reader for the ring.
    pub fn ring_reader(&self, ring: usize, from: ReadFrom) -> Result<RingReader, Error> {
        let desc = *self.inner.rings.get(ring).ok_or(Error::NoSuchRing)?;
        Ok(self.reader_on(desc, from, None))
    }

    /// Records written so far to the ring (its head): the next record's sequence number.
    pub fn ring_head(&self, ring: usize) -> Result<u64, Error> {
        let desc = *self.inner.rings.get(ring).ok_or(Error::NoSuchRing)?;
        Ok(self.ring_ctrl(&desc).head.load(Ordering::Acquire))
    }

    /// The reserved ring region, as (offset from the arena's start, length), both multiples of [`PAGE_ALIGN`].
    pub fn ring_region(&self) -> (u64, u64) {
        self.inner.ring_region
    }

    /// Carves a ring of `capacity` records of `stride` bytes at `offset` in the ring region (a multiple of
    /// [`PAGE_ALIGN`]; it takes [`CarvedRing::span`] bytes from there). The block must never have been carved and
    /// must not overlap a ring carved before (checked against the superblock's table): the region starts zeroed, so
    /// head and started start at 0 on memory no reader has seen. The block is taken with a compare-and-swap on its
    /// generation, so two carvers can't both win it, and the ring is recorded in the table (its index there is the
    /// ring's index). Only the arena's owner (the broker) carves.
    ///
    /// # Errors
    /// [`Error::NoSuchRing`] if the ring doesn't fit the region there; [`Error::BadConfig`] if that block is in use,
    /// overlaps another ring, or the table is full.
    pub fn carve_ring(&self, offset: u64, stride: u32, capacity: u32) -> Result<CarvedRing, Error> {
        let ring = CarvedRing {
            ctrl_offset: offset,
            data_offset: offset.checked_add(PAGE_ALIGN as u64).ok_or(Error::NoSuchRing)?,
            stride,
            capacity,
            generation: 1,
        };
        let desc = self.carved_desc(&ring)?;
        // Before touching the block: it must not lie in another ring's pages.
        let end = ring.data_offset + desc.data_len;
        if let Some(other) = self
            .carved_rings()
            .into_iter()
            .find(|o| ring.ctrl_offset < o.data_offset + o.data_len() && o.ctrl_offset < end)
        {
            return Err(Error::BadConfig(format!(
                "a ring at {offset} would overlap the ring at {}",
                other.ctrl_offset
            )));
        }
        let ctrl = self.ring_ctrl(&desc);
        let untouched = ctrl.head.load(Ordering::SeqCst) == 0
            && ctrl.started.load(Ordering::SeqCst) == 0
            && ctrl.writer.load(Ordering::SeqCst) == 0;
        // Taking the generation from 0 is what claims the block; it's announced (and readable) only after.
        if !untouched
            || ctrl.generation.compare_exchange(0, ring.generation, Ordering::SeqCst, Ordering::SeqCst).is_err()
        {
            return Err(Error::BadConfig(format!("the ring region at {offset} is in use")));
        }
        self.append_carved(&ring)?;
        Ok(ring)
    }

    /// Takes a carved ring's single writer. [`Error::RingBusy`] if another writer holds it; [`Error::Stale`] if the
    /// block was carved again since `ring` was announced.
    pub fn carved_ring_writer(&self, ring: &CarvedRing) -> Result<RingWriter, Error> {
        let desc = self.carved(ring)?;
        let token = claim_token();
        self.ring_ctrl(&desc)
            .writer
            .compare_exchange(0, token, Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|_| Error::RingBusy(ring.ctrl_offset as usize))?;
        Ok(RingWriter { arena: self.clone(), desc, generation: Some(ring.generation), token })
    }

    /// Makes a reader for a carved ring. [`Error::Stale`] if the block was carved again since `ring` was announced.
    pub fn carved_ring_reader(&self, ring: &CarvedRing, from: ReadFrom) -> Result<RingReader, Error> {
        let desc = self.carved(ring)?;
        Ok(self.reader_on(desc, from, Some(ring.generation)))
    }

    /// Records written so far to a carved ring.
    pub fn carved_ring_head(&self, ring: &CarvedRing) -> Result<u64, Error> {
        let desc = self.carved(ring)?;
        Ok(self.ring_ctrl(&desc).head.load(Ordering::Acquire))
    }

    /// Lets another writer take a carved ring whose writer is gone without dropping its [`RingWriter`] (its process
    /// died). The broker calls it when the writer's session ends. The old writer, if it's still running, gets
    /// [`Error::NotWriter`] from its next push, and its `Drop` leaves the next writer's claim alone.
    pub fn release_carved_ring_writer(&self, ring: &CarvedRing) -> Result<(), Error> {
        let desc = self.carved(ring)?;
        self.ring_ctrl(&desc).writer.store(0, Ordering::SeqCst);
        Ok(())
    }

    /// Releases a carved ring's writer claim only if process `pid` holds it (a claim token carries its process id in
    /// its high half). The broker's release when a session ends, so a late close of an old session can't clear a
    /// claim a newer writer took since. Returns whether it released one.
    ///
    /// # Errors
    /// As [`Arena::carved_ring_writer`].
    pub fn release_carved_ring_writer_of(&self, ring: &CarvedRing, pid: u32) -> Result<bool, Error> {
        let desc = self.carved(ring)?;
        let writer = &self.ring_ctrl(&desc).writer;
        let mut word = writer.load(Ordering::SeqCst);
        loop {
            if word == 0 || word >> 32 != u64::from(pid) {
                return Ok(false);
            }
            match writer.compare_exchange(word, 0, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => return Ok(true),
                Err(w) => word = w,
            }
        }
    }

    fn reader_on(&self, desc: RingDesc, from: ReadFrom, generation: Option<u32>) -> RingReader {
        let head = self.ring_ctrl(&desc).head.load(Ordering::Acquire);
        let cursor = match from {
            ReadFrom::Latest => head,
            ReadFrom::Oldest => head.saturating_sub(u64::from(desc.capacity)),
        };
        RingReader { arena: self.clone(), desc, cursor, pending_lost: 0, generation }
    }

    /// A carved ring's descriptor, checked against the control block's generation and against the superblock's
    /// own record of the ring, so an announcement that doesn't match what was carved (a stale `Subscribed`, a broker
    /// bug) can't point a reader into another ring's records.
    fn carved(&self, ring: &CarvedRing) -> Result<RingDesc, Error> {
        let desc = self.carved_desc(ring)?;
        if self.ring_ctrl(&desc).generation.load(Ordering::Acquire) != ring.generation || ring.generation == 0 {
            return Err(Error::Stale);
        }
        if !self.carved_rings().contains(ring) {
            return Err(Error::NoSuchRing);
        }
        Ok(desc)
    }

    /// A carved ring's descriptor, its numbers checked against the ring region: never trusted raw.
    fn carved_desc(&self, ring: &CarvedRing) -> Result<RingDesc, Error> {
        let page = PAGE_ALIGN as u64;
        let (start, len) = self.inner.ring_region;
        let end = start + len;
        let data_len = u64::from(ring.stride)
            .checked_mul(u64::from(ring.capacity))
            .and_then(|n| n.checked_next_multiple_of(page))
            .ok_or(Error::NoSuchRing)?;
        let ctrl_end = ring.ctrl_offset.checked_add(size_of::<RingCtrl>() as u64).ok_or(Error::NoSuchRing)?;
        let data_end = ring.data_offset.checked_add(data_len).ok_or(Error::NoSuchRing)?;
        // Exactly the layout carve_ring makes: a control page, then the records on the next page.
        let ok = ring.stride > 0
            && ring.capacity > 0
            && ring.ctrl_offset.is_multiple_of(page)
            && ring.ctrl_offset.checked_add(page) == Some(ring.data_offset)
            && ring.data_offset.is_multiple_of(page)
            && ring.ctrl_offset >= start
            && ctrl_end <= end
            && ring.data_offset >= start
            && data_end <= end
            && (ctrl_end <= ring.data_offset || data_end <= ring.ctrl_offset);
        if !ok {
            return Err(Error::NoSuchRing);
        }
        Ok(RingDesc {
            stride: ring.stride,
            capacity: ring.capacity,
            ctrl_offset: ring.ctrl_offset,
            data_offset: ring.data_offset,
            data_len,
        })
    }
}

/// Whether a record at `ptr` of `len` bytes is copied in 8-byte words (else bytes).
fn by_words(ptr: *const u8, len: usize) -> bool {
    (ptr as usize).is_multiple_of(8) && len.is_multiple_of(8)
}

/// Writes `src` to shared memory at `dst` with relaxed atomic stores.
///
/// # Safety
///
/// `dst` must be valid for `src.len()` bytes, and every other access to them must go through [`store_record`] or
/// [`load_record`] with the same pointer and length.
unsafe fn store_record(dst: *mut u8, src: &[u8]) {
    if by_words(dst, src.len()) {
        for (i, chunk) in src.as_chunks::<8>().0.iter().enumerate() {
            let word = u64::from_ne_bytes(*chunk);
            // SAFETY: in bounds and 8-byte aligned (by_words), and only ever accessed atomically.
            unsafe { AtomicU64::from_ptr(dst.add(i * 8).cast()) }.store(word, Ordering::Relaxed);
        }
    } else {
        for (i, &byte) in src.iter().enumerate() {
            // SAFETY: in bounds, and only ever accessed atomically.
            unsafe { AtomicU8::from_ptr(dst.add(i)) }.store(byte, Ordering::Relaxed);
        }
    }
}

/// Reads `dst.len()` bytes of shared memory at `src` with relaxed atomic loads.
///
/// # Safety
///
/// As for [`store_record`].
unsafe fn load_record(src: *mut u8, dst: &mut [u8]) {
    if by_words(src, dst.len()) {
        for (i, chunk) in dst.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            // SAFETY: in bounds and 8-byte aligned (by_words), and only ever accessed atomically.
            let word = unsafe { AtomicU64::from_ptr(src.add(i * 8).cast()) }.load(Ordering::Relaxed);
            chunk.copy_from_slice(&word.to_ne_bytes());
        }
    } else {
        for (i, byte) in dst.iter_mut().enumerate() {
            // SAFETY: in bounds, and only ever accessed atomically.
            *byte = unsafe { AtomicU8::from_ptr(src.add(i)) }.load(Ordering::Relaxed);
        }
    }
}

fn record_ptr(arena: &Arena, desc: &RingDesc, seq: u64) -> *mut u8 {
    let slot = (seq % u64::from(desc.capacity)) as usize;
    arena.ptr_at(desc.data_offset as usize + slot * desc.stride as usize)
}

impl RingCtrl {
    fn seq_wait(&self) -> SeqWait<'_> {
        SeqWait::new(&self.wait_word, &self.waiters)
    }
}

/// The ring's writer. Dropping it lets another writer take the ring.
pub struct RingWriter {
    arena: Arena,
    desc: RingDesc,
    /// A carved ring's generation, checked on every push.
    generation: Option<u32>,
    /// This claim's token, in the control block's `writer` while the claim holds.
    token: u64,
}

impl RingWriter {
    /// Bytes per record.
    pub fn stride(&self) -> usize {
        self.desc.stride as usize
    }

    /// Appends a record of exactly `stride()` bytes, overwriting the oldest if the ring is full. Returns its
    /// sequence number.
    pub fn push(&mut self, record: &[u8]) -> Result<u64, Error> {
        if record.len() != self.stride() {
            return Err(Error::BadRecord { got: record.len(), stride: self.stride() });
        }
        let ctrl = self.ctrl();
        if let Some(g) = self.generation
            && ctrl.generation.load(Ordering::Acquire) != g
        {
            return Err(Error::Stale);
        }
        if ctrl.writer.load(Ordering::Acquire) != self.token {
            return Err(Error::NotWriter);
        }
        let seq = ctrl.head.load(Ordering::Relaxed); // only this writer changes head and started
        ctrl.started.store(seq + 1, Ordering::Relaxed);
        // A reader whose copy sees any of the writes below also sees `started` at least `seq + 1`, and so rejects
        // the record `capacity` back that shares this slot.
        fence(Ordering::Release);
        // SAFETY: the record slot lies inside the ring's segment and holds stride bytes, and only this writer writes
        // it. Readers may copy it concurrently, through atomics of the same width.
        unsafe { store_record(record_ptr(&self.arena, &self.desc, seq), record) };
        // Checked again before publishing: a writer released mid-record doesn't publish it. What this doesn't close:
        // an old writer descheduled inside store_record (or between this check and the store below) can land its
        // stray bytes after the new writer has taken the ring, so a reader can validate a torn record in that slot,
        // or a later record there if the old writer stalls past a lap. Readers of fixed-stride streams check each
        // record's own continuity (sequence or stamp) and treat a break as a gap; the next push stops the old writer.
        if ctrl.writer.load(Ordering::Acquire) != self.token {
            return Err(Error::NotWriter);
        }
        ctrl.head.store(seq + 1, Ordering::Release);
        ctrl.seq_wait().publish((seq + 1) as u32);
        Ok(seq)
    }

    /// Records written so far.
    pub fn head(&self) -> u64 {
        self.ctrl().head.load(Ordering::Relaxed)
    }

    fn ctrl(&self) -> &RingCtrl {
        self.arena.ring_ctrl(&self.desc)
    }
}

impl Drop for RingWriter {
    fn drop(&mut self) {
        // Only this claim's own token is cleared: if the broker released the ring and another writer took it, that
        // writer's claim stays.
        let _ = self.ctrl().writer.compare_exchange(self.token, 0, Ordering::SeqCst, Ordering::SeqCst);
    }
}

/// A reader with its own cursor.
pub struct RingReader {
    arena: Arena,
    desc: RingDesc,
    cursor: u64,
    /// Records skipped by a read that then found nothing to return, reported with the next record.
    pending_lost: u64,
    /// A carved ring's generation, checked around every read.
    generation: Option<u32>,
}

/// What a blocked reader needs, apart from the reader itself, so a caller can wait without holding the reader (the
/// Python bindings wait with the GIL released while other threads keep reading).
pub struct RingWaiter {
    arena: Arena,
    desc: RingDesc,
    cursor: u64,
}

impl RingWaiter {
    /// Blocks until a record after the reader's cursor is published, or `timeout` passes, without polling. Returns
    /// whether one is ready.
    pub fn wait(&self, timeout: Duration) -> bool {
        let deadline = crate::seqwait::deadline_after(timeout);
        let ctrl = self.arena.ring_ctrl(&self.desc);
        let seq_wait = ctrl.seq_wait();
        loop {
            let seen = seq_wait.load();
            if ctrl.head.load(Ordering::Acquire) > self.cursor {
                return true;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return false;
            }
            seq_wait.wait(seen, left);
        }
    }
}

impl RingReader {
    /// Bytes per record.
    pub fn stride(&self) -> usize {
        self.desc.stride as usize
    }

    /// The sequence number of the next record this reader will read.
    pub fn cursor(&self) -> u64 {
        self.cursor
    }

    /// Blocks until a record this reader hasn't read is published, or `timeout` passes, without polling. Returns
    /// whether one is ready.
    pub fn wait(&self, timeout: Duration) -> bool {
        self.waiter().wait(timeout)
    }

    /// A snapshot of this reader's position that can wait on its own (see [`RingWaiter`]).
    pub fn waiter(&self) -> RingWaiter {
        RingWaiter { arena: self.arena.clone(), desc: self.desc, cursor: self.cursor }
    }

    /// Moves the reader to record `seq`, so the next [`RingReader::read`] gives it (or, if the ring has lapped it
    /// already, the oldest record still there, with the skipped ones counted as lost). For a reader that follows
    /// batches a notice names by sequence, like the tier worker.
    pub fn seek(&mut self, seq: u64) {
        self.cursor = seq;
        self.pending_lost = 0;
    }

    /// The sequence number of the record the next read gives (before any lapping).
    #[must_use]
    pub fn position(&self) -> u64 {
        self.cursor
    }

    /// Copies the next record into `out` (exactly `stride()` bytes). `None` if the reader is caught up.
    pub fn read(&mut self, out: &mut [u8]) -> Result<Option<Record>, Error> {
        if out.len() != self.stride() {
            return Err(Error::BadRecord { got: out.len(), stride: self.stride() });
        }
        let capacity = u64::from(self.desc.capacity);
        let ctrl = self.arena.ring_ctrl(&self.desc);
        let current = |ctrl: &RingCtrl| self.generation.is_none_or(|g| ctrl.generation.load(Ordering::Acquire) == g);
        if !current(ctrl) {
            return Err(Error::Stale);
        }
        let mut lost = std::mem::take(&mut self.pending_lost);
        loop {
            let head = ctrl.head.load(Ordering::Acquire);
            if self.cursor >= head {
                self.pending_lost = lost;
                return Ok(None);
            }
            let oldest = head.saturating_sub(capacity);
            if self.cursor < oldest {
                lost += oldest - self.cursor;
                self.cursor = oldest;
            }
            let seq = self.cursor;
            // SAFETY: the slot lies inside the ring's segment and holds stride bytes (out.len()). The writer may be
            // overwriting it through atomics of the same width; the recheck below discards such a copy.
            unsafe { load_record(record_ptr(&self.arena, &self.desc, seq), out) };
            fence(Ordering::Acquire);
            let started = ctrl.started.load(Ordering::Relaxed);
            // A copy from a block carved again since is someone else's record.
            if !current(ctrl) {
                return Err(Error::Stale);
            }
            // Record `seq + capacity` reuses this slot; the copy is good if the writer hadn't begun it.
            if started <= seq + capacity {
                self.cursor = seq + 1;
                return Ok(Some(Record { seq, lost }));
            }
            let skip_to = started - capacity;
            lost += skip_to - seq;
            self.cursor = skip_to;
        }
    }
}
