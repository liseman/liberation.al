//! The Tier 0 arena: the shared memory the broker creates at start, passes to every client with `Welcome`, and
//! checks Tier 0 locators against.
//!
//! One arena per broker, sized from the machine profile's slot classes. Its name is `/cnt.<h>.<generation>`, where
//! `<h>` is a hash of the socket path (two brokers on one machine, like parallel tests, never share an arena) and the
//! generation is the later of one more than the last run's (kept in `<socket>.arena`) and the clock's seconds, so it
//! keeps rising even if that file is lost: an old run's Tier 0 locator never names the new arena. A broker that crashed leaves its arena
//! behind; the next one on the same socket unlinks it first. The broker holds the socket's lock before any of this,
//! so a second broker on the same socket never gets this far.
//!
//! Producers claim and publish slots themselves, in shared memory (see `continuum-shm`); the broker only checks an
//! `IngestNotice`'s or a commit's locator: the arena and its generation are this one, the slot exists at that offset,
//! and it's published at the locator's slot generation with the locator's length.

use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::time::Duration;

use std::collections::HashMap;
use std::sync::Mutex;

use continuum_core::{SlotHandle, StorageLocator, StreamId};
use continuum_sched::SlotClass;
use continuum_shm::{Arena, ArenaConfig, CarvedRing, SlabClass};
use tracing::{info, warn};

/// How long the check's own pin on a slot may last. It's released as soon as the check is done.
const CHECK_PIN: Duration = Duration::from_millis(100);

/// How long a query result's pin lasts: the reader has this long to read the slot (or renew the pin).
pub const READ_PIN: Duration = Duration::from_secs(5);

/// What `Welcome` says about an arena.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArenaSummary {
    /// The shared memory name.
    pub name: String,
    /// The arena's generation.
    pub generation: u32,
    /// The arena's whole size in bytes.
    pub size_bytes: u64,
    /// Each class's slot size and count, in class order.
    pub classes: Vec<(u64, u32)>,
}

/// How long a ring keeps its records when `RegisterStream` asks for no particular time.
pub const DEFAULT_RING_SECONDS: f32 = 10.0;

/// A ring carved for a stream, and the session that registered it last (its writer).
#[derive(Clone, Copy, Debug)]
struct StreamRing {
    stream: StreamId,
    ring: CarvedRing,
}

/// The ring region's bookkeeping: rings are carved in turn and kept for the arena's life (one per stream, the same
/// one again when the stream registers again).
#[derive(Debug, Default)]
struct Rings {
    /// Where the next ring goes, from the region's start.
    next: u64,
    /// By index: what `RingLocation.ring` names.
    rings: Vec<StreamRing>,
    by_stream: HashMap<StreamId, u16>,
}

/// The broker's arena, and the fd every `Welcome` passes for it.
pub struct Tier0 {
    arena: Arena,
    fd: OwnedFd,
    summary: ArenaSummary,
    rings: Mutex<Rings>,
    /// Whether the arena was made with `reclaim_requires_flush`, so a slot's `flushed` mark means a Tier 1 copy.
    requires_flush: bool,
}

impl std::fmt::Debug for Tier0 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tier0").field("summary", &self.summary).finish_non_exhaustive()
    }
}

/// The file beside the socket that keeps the last arena generation: `<socket>.arena`.
#[must_use]
pub fn generation_path(socket: &Path) -> PathBuf {
    let mut name = socket.as_os_str().to_owned();
    name.push(".arena");
    PathBuf::from(name)
}

/// The arena name for `socket` at `generation`.
#[must_use]
pub fn arena_name(socket: &Path, generation: u32) -> String {
    // 48 bits of a 64-bit FNV-1a: wide enough that two sockets on one machine practically never share a name (the
    // name is replaced on start if it exists), and `/cnt.` + 12 + `.` + 10 digits stays within macOS's 31 bytes.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in socket.as_os_str().as_encoded_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("/cnt.{:012x}.{generation}", h >> 16)
}

/// The broker's arena always waits for the tier worker's `flushed` mark before an expired pin's slot is reused.
const RECLAIM_REQUIRES_FLUSH: bool = true;

impl Tier0 {
    /// Creates the arena for the broker on `socket`, with `classes`, after unlinking the last run's.
    ///
    /// # Errors
    /// The generation file can't be read or written, or the arena can't be created or reopened for its fd.
    pub fn create(socket: &Path, classes: &[SlotClass], ring_bytes: u64) -> Result<Self, String> {
        let file = generation_path(socket);
        let last = match std::fs::read_to_string(&file) {
            Ok(s) => s.lines().next().and_then(|g| g.trim().parse::<u32>().ok()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(format!("reading {}: {e}", file.display())),
        };
        if let Some(g) = last {
            unlink_quietly(&arena_name(socket, g));
        }
        // Rising across restarts even without the file; never 0, so a default locator never names a real arena.
        let clock = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| u32::try_from(d.as_secs()).unwrap_or(u32::MAX));
        let generation = last.map_or(1, |g| g.saturating_add(1)).max(clock).max(1);
        let name = arena_name(socket, generation);
        // Written before the arena exists, so a crash between the two still leaves the name for the next start. The
        // second line is the name itself, for whoever has to clear an arena a killed broker left (see the doc).
        std::fs::write(&file, format!("{generation}\n{name}\n"))
            .map_err(|e| format!("writing {}: {e}", file.display()))?;
        let config = ArenaConfig {
            classes: classes
                .iter()
                .map(|c| {
                    Ok(SlabClass {
                        slot_size: usize::try_from(c.size_bytes).map_err(|_| "a slot class is too large".to_owned())?,
                        slot_count: c.count,
                    })
                })
                .collect::<Result<_, String>>()?,
            rings: Vec::new(),
            generation,
            // The tier worker marks slots flushed once their chunk is committed (continuum-tier's service), so an
            // expired pin no longer lets a slot be reused before its Tier 1 copy exists.
            reclaim_requires_flush: RECLAIM_REQUIRES_FLUSH,
            ring_region_bytes: usize::try_from(ring_bytes).map_err(|_| "the ring region is too large".to_owned())?,
        };
        // A name left by a run whose generation file was lost: it's this socket's, so it's safe to replace.
        let arena = match Arena::create(&name, &config) {
            Ok(a) => a,
            Err(continuum_shm::Error::Io(e)) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                unlink_quietly(&name);
                Arena::create(&name, &config).map_err(|e| format!("creating arena {name}: {e}"))?
            }
            Err(e) => return Err(format!("creating arena {name}: {e}")),
        };
        if let Err(e) = check_room(arena.len() as u64) {
            drop(arena);
            unlink_quietly(&name);
            return Err(e);
        }
        let fd = rustix::shm::open(name.as_str(), rustix::shm::OFlags::RDWR, rustix::fs::Mode::empty())
            .map_err(|e| format!("reopening arena {name} for its fd: {e}"))?;
        let classes = (0..arena.class_count())
            .filter_map(|i| u16::try_from(i).ok())
            .filter_map(|i| arena.class(i).ok())
            .map(|c| (c.slot_size as u64, c.slot_count))
            .collect();
        let summary = ArenaSummary { name, generation, size_bytes: arena.len() as u64, classes };
        info!(arena = %summary.name, size_bytes = summary.size_bytes, "created the Tier 0 arena");
        Ok(Self { arena, fd, summary, rings: Mutex::new(Rings::default()), requires_flush: RECLAIM_REQUIRES_FLUSH })
    }

    /// What `Welcome` lists for this arena.
    #[must_use]
    pub fn summary(&self) -> &ArenaSummary {
        &self.summary
    }

    /// The fd `Welcome` passes for the arena.
    #[must_use]
    pub fn fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }

    /// Checks a locator an ingest or a commit names. Only a Tier 0 locator needs checking here; the others pass.
    ///
    /// # Errors
    /// Why the locator doesn't name a published slot or ring batch of this arena, as the rejection's reason.
    pub fn check(&self, locator: &StorageLocator) -> Result<(), String> {
        match locator {
            StorageLocator::Shm { .. } => self.pin(locator, CHECK_PIN).map(drop),
            StorageLocator::Ring { arena, arena_generation, ring, first_seq, count } => {
                self.this_arena(arena, *arena_generation)?;
                let carved = self.ring(*ring).ok_or_else(|| format!("ring {ring} doesn't exist"))?;
                let head = self.arena.carved_ring_head(&carved).map_err(|e| format!("ring {ring}: {e}"))?;
                let end = first_seq.checked_add(u64::from(*count)).ok_or("the batch's sequence numbers overflow")?;
                if *count == 0 {
                    return Err("an empty ring batch".to_owned());
                }
                if end > head {
                    return Err(format!("ring {ring} has {head} records; the batch ends at {end}"));
                }
                let oldest = head.saturating_sub(u64::from(carved.capacity));
                if *first_seq < oldest {
                    return Err(format!(
                        "ring {ring} has lapped the batch: its oldest record is {oldest}, the batch starts at {first_seq}"
                    ));
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// [`Tier0::check`], plus that a ring batch is in `stream`'s own ring.
    ///
    /// # Errors
    /// As [`Tier0::check`], or the ring isn't the stream's.
    pub fn check_for(&self, stream: StreamId, locator: &StorageLocator) -> Result<(), String> {
        if let StorageLocator::Ring { ring, .. } = locator
            && self.ring_of(stream).map(|(index, _)| index) != Some(*ring)
        {
            return Err(format!("ring {ring} isn't stream {}'s ring", stream.0));
        }
        self.check(locator)
    }

    fn this_arena(&self, arena: &str, generation: u32) -> Result<(), String> {
        if arena != self.summary.name || generation != self.summary.generation {
            return Err(format!(
                "Tier 0 locator names arena {arena} generation {generation}; this broker's is {} generation {}",
                self.summary.name, self.summary.generation
            ));
        }
        Ok(())
    }

    /// A ring's head (records written so far) and capacity, for the sweep of lapped batches.
    #[must_use]
    pub fn ring_head(&self, index: u16) -> Option<(u64, u32)> {
        let ring = self.ring(index)?;
        self.arena.carved_ring_head(&ring).ok().map(|head| (head, ring.capacity))
    }

    /// Whether the arena was made with `reclaim_requires_flush`. Without it, `flushed` carries no archive promise,
    /// and the scheduler's reclaim treats every slot as it treats one with no Tier 1 obligation.
    #[must_use]
    pub fn reclaim_requires_flush(&self) -> bool {
        self.requires_flush
    }

    /// How many slot classes the arena has.
    #[must_use]
    pub fn class_count(&self) -> u16 {
        u16::try_from(self.arena.class_count()).unwrap_or(u16::MAX)
    }

    /// Every slot in `class` as it is now (see [`continuum_shm::Arena::scan`]).
    ///
    /// # Errors
    /// The class doesn't exist.
    pub fn scan(&self, class: u16) -> Result<Vec<continuum_shm::SlotScan>, continuum_shm::Error> {
        self.arena.scan(class)
    }

    /// Starts reclaiming a published slot (see [`continuum_shm::Arena::reclaim`]).
    ///
    /// # Errors
    /// The slot already moved on.
    pub fn reclaim(&self, handle: SlotHandle) -> Result<continuum_shm::Reclaimed, continuum_shm::Error> {
        self.arena.reclaim(handle)
    }

    /// Takes a purged payload's slot back and zeroes it (see [`continuum_shm::Arena::purge`]).
    ///
    /// # Errors
    /// The slot already moved on.
    pub fn purge_slot(&self, handle: SlotHandle) -> Result<continuum_shm::Purged, continuum_shm::Error> {
        self.arena.purge(handle)
    }

    /// `stream`'s ring, if it has one: its index and layout.
    #[must_use]
    pub fn ring_of(&self, stream: StreamId) -> Option<(u16, CarvedRing)> {
        let rings = self.rings.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let index = *rings.by_stream.get(&stream)?;
        Some((index, rings.rings[usize::from(index)].ring))
    }

    /// The ring at `index`, as it was carved.
    #[must_use]
    pub fn ring(&self, index: u16) -> Option<CarvedRing> {
        let rings = self.rings.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        rings.rings.get(usize::from(index)).map(|r| r.ring)
    }

    /// The ring for `stream`, `writer`'s from now on: the one it has (when the stride matches), or a new one of
    /// `stride` bytes per record holding `seconds` (0 for [`DEFAULT_RING_SECONDS`]) at `rate_hz` records a second.
    /// Returns its index and layout.
    ///
    /// # Errors
    /// The stride or rate can't make a ring, the stream already has a ring of another stride, or the ring region
    /// has no room left.
    pub fn ring_for(
        &self,
        stream: StreamId,
        stride: u32,
        rate_hz: f64,
        seconds: f32,
    ) -> Result<(u16, CarvedRing), String> {
        let mut rings = self.rings.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(&index) = rings.by_stream.get(&stream) {
            let r = &mut rings.rings[usize::from(index)];
            if r.ring.stride != stride {
                return Err(format!("the stream already has a ring of {} byte records, not {stride}", r.ring.stride));
            }
            return Ok((index, r.ring));
        }
        let seconds = if seconds > 0.0 { f64::from(seconds) } else { f64::from(DEFAULT_RING_SECONDS) };
        let records = (seconds * rate_hz).ceil();
        if stride == 0 || !(1.0..=f64::from(u32::MAX)).contains(&records) {
            return Err(format!("a ring of {stride} byte records for {seconds} s at {rate_hz} Hz can't be made"));
        }
        let capacity = records as u32;
        let span = CarvedRing::span(stride, capacity).ok_or("the ring is too large")?;
        let (start, len) = self.arena.ring_region();
        let left = len.saturating_sub(rings.next);
        if span > left {
            return Err(format!(
                "the ring needs {span} bytes and the ring region has {left} left ({len} in all; see --tier0-ring-mib)"
            ));
        }
        let index = u16::try_from(rings.rings.len()).map_err(|_| "too many rings".to_owned())?;
        let ring = self.arena.carve_ring(start + rings.next, stride, capacity).map_err(|e| e.to_string())?;
        rings.next += span;
        rings.rings.push(StreamRing { stream, ring });
        rings.by_stream.insert(stream, index);
        info!(stream = stream.0, index, stride, capacity, "carved a ring");
        Ok((index, ring))
    }

    /// Releases every ring claim process `pid` holds: the process whose session just ended (and that has no other
    /// session open), so a stream's next writer can take its ring even if this one died holding it. A claim's token
    /// carries its process id, so whichever ring the process actually claimed is released, and no other process's
    /// claim is touched, however late the close.
    pub fn release_rings_of_pid(&self, pid: u32) {
        let rings = self.rings.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        for r in &rings.rings {
            if let Err(e) = self.arena.release_carved_ring_writer_of(&r.ring, pid) {
                warn!(stream = r.stream.0, error = %e, "couldn't release a ring's writer");
            }
        }
    }

    /// Pins the slot a Tier 0 locator names, for `ttl`, after the same checks as [`Tier0::check`]: the pin a query
    /// result hands a reader. Dropping the returned lease releases it; until then (or until `ttl` passes) the slot's
    /// bytes don't change.
    ///
    /// # Errors
    /// The locator isn't Tier 0, or doesn't name a published slot of this arena (stale or reused: the caller drops
    /// the item, or answers from another copy).
    pub fn pin(&self, locator: &StorageLocator, ttl: Duration) -> Result<continuum_shm::Lease, String> {
        let StorageLocator::Shm { arena, arena_generation, slab_class, slot, slot_generation, offset, length } =
            locator
        else {
            return Err("not a Tier 0 locator".to_owned());
        };
        self.this_arena(arena, *arena_generation)?;
        let at = self
            .arena
            .slot_offset(*slab_class, *slot)
            .map_err(|_| format!("Tier 0 locator names slot {slot} of class {slab_class}, which doesn't exist"))?;
        if at as u64 != *offset {
            return Err(format!("Tier 0 locator's offset is {offset}; slot {slot} of class {slab_class} is at {at}"));
        }
        let handle = SlotHandle { class: *slab_class, index: *slot, generation: *slot_generation };
        // A pin proves the slot is published at that generation.
        let pin = self.arena.lease(handle, ttl).map_err(|e| {
            format!("Tier 0 slot {slot} of class {slab_class} at generation {slot_generation} isn't published: {e}")
        })?;
        if pin.len() as u64 != *length {
            return Err(format!("Tier 0 locator's length is {length}; the slot holds {}", pin.len()));
        }
        Ok(pin)
    }
}

impl Tier0 {
    /// Removes the arena's name, so nothing new can open it; every mapping (the broker's, its clients') stays
    /// valid. The server does this when it stops, without waiting for the scheduler task to let go of the arena.
    pub fn unlink(&self) {
        unlink_quietly(&self.summary.name);
    }
}

impl Drop for Tier0 {
    fn drop(&mut self) {
        self.unlink();
    }
}

/// How many bytes the shared memory filesystem holds in all (`/dev/shm` on Linux), or `None` where an arena has no
/// such size to fit (macOS, or a Linux without `/dev/shm`).
#[must_use]
pub fn shm_capacity() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let fs = rustix::fs::statvfs("/dev/shm").ok()?;
        Some(fs.f_blocks.saturating_mul(fs.f_frsize))
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Fails when the arena is larger than the shared memory filesystem (`/dev/shm` on Linux), where touching its last
/// pages would SIGBUS the broker, and warns when it's larger than what's free there now. macOS has no such limit to
/// check (its POSIX shared memory is backed by the VM system).
#[cfg(target_os = "linux")]
fn check_room(size: u64) -> Result<(), String> {
    let Ok(fs) = rustix::fs::statvfs("/dev/shm") else { return Ok(()) };
    let total = fs.f_blocks.saturating_mul(fs.f_frsize);
    let free = fs.f_bavail.saturating_mul(fs.f_frsize);
    if size > total {
        return Err(format!(
            "the Tier 0 arena needs {size} bytes but /dev/shm holds {total}; leave --profile and --tier0-classes \
             unset (the default fits itself to /dev/shm), or give /dev/shm more room"
        ));
    }
    if size > free {
        warn!(size, free, "the Tier 0 arena is larger than what's free in /dev/shm now; it fills as slots are used");
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn check_room(_size: u64) -> Result<(), String> {
    Ok(())
}

fn unlink_quietly(name: &str) {
    match Arena::unlink(name) {
        Ok(()) => info!(arena = name, "unlinked a Tier 0 arena"),
        Err(continuum_shm::Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => warn!(arena = name, error = %e, "couldn't unlink a Tier 0 arena"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tier0(dir: &Path) -> Tier0 {
        Tier0::create(&dir.join("b.sock"), &[SlotClass { size_bytes: 16_384, count: 2 }], 1 << 20).unwrap()
    }

    fn published(t: &Tier0, len: usize) -> StorageLocator {
        let mut slot = t.arena.claim(0, 1).unwrap();
        slot.as_mut_slice()[..len].fill(1);
        let offset = slot.offset() as u64;
        let h = slot.publish(len).unwrap();
        StorageLocator::Shm {
            arena: t.summary.name.clone(),
            arena_generation: t.summary.generation,
            slab_class: h.class,
            slot: h.index,
            slot_generation: h.generation,
            offset,
            length: len as u64,
        }
    }

    fn handle(l: &StorageLocator) -> SlotHandle {
        let StorageLocator::Shm { slab_class, slot, slot_generation, .. } = l else { panic!() };
        SlotHandle { class: *slab_class, index: *slot, generation: *slot_generation }
    }

    #[test]
    fn a_pin_holds_the_slot_until_dropped_and_a_check_leaves_none() {
        let dir = tempfile::tempdir().unwrap();
        let t = tier0(dir.path());
        let loc = published(&t, 10);
        t.check(&loc).unwrap();
        assert_eq!(t.arena.pin_count(handle(&loc)).unwrap(), 0, "a check's pin is released");
        let pin = t.pin(&loc, READ_PIN).unwrap();
        assert_eq!(t.arena.pin_count(handle(&loc)).unwrap(), 1);
        assert_eq!(pin.len(), 10);
        // Pinned, the slot can't be reclaimed and reused.
        assert!(t.arena.reclaim(handle(&loc)).is_ok());
        assert_eq!(t.arena.pin_count(handle(&loc)).unwrap(), 1, "still pinned while reclaiming");
        drop(pin);
        assert_eq!(t.arena.pin_count(handle(&loc)).unwrap(), 0);
    }

    #[test]
    fn a_stale_or_non_tier_0_locator_isnt_pinned() {
        let dir = tempfile::tempdir().unwrap();
        let t = tier0(dir.path());
        let loc = published(&t, 10);
        t.arena.reclaim(handle(&loc)).unwrap(); // reused from here on
        assert!(t.pin(&loc, READ_PIN).err().unwrap().contains("isn't published"));
        let inline = StorageLocator::inline(vec![1, 2, 3]).unwrap();
        assert!(t.pin(&inline, READ_PIN).is_err());
        assert!(t.check(&inline).is_ok(), "other locators pass the check");
    }

    #[test]
    fn a_closing_process_releases_only_its_own_claim_whichever_session_registered_the_ring() {
        let dir = tempfile::tempdir().unwrap();
        let t = tier0(dir.path());
        let mic = StreamId(7);
        let (index, ring) = t.ring_for(mic, 64, 100.0, 1.0).unwrap();
        // Registering again gets the same ring back.
        assert_eq!(t.ring_for(mic, 64, 100.0, 1.0).unwrap(), (index, ring));
        // This process claims it, then dies without dropping its writer.
        std::mem::forget(t.arena.carved_ring_writer(&ring).unwrap());
        // Another process's close releases nothing; this one's releases the claim.
        t.release_rings_of_pid(std::process::id().wrapping_add(1));
        assert!(t.arena.carved_ring_writer(&ring).is_err());
        t.release_rings_of_pid(std::process::id());
        assert!(t.arena.carved_ring_writer(&ring).is_ok(), "released when its process's session ended");
    }

    #[test]
    fn dropping_the_arena_unlinks_it() {
        let dir = tempfile::tempdir().unwrap();
        let name = tier0(dir.path()).summary.name.clone();
        assert!(Arena::open(&name).is_err());
    }
}
