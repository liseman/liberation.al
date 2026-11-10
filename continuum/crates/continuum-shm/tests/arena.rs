//! The arena against real POSIX shared memory: alignment, claims, generations, pins, lease expiry, and rings.

use std::os::fd::{AsFd, FromRawFd};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread;
use std::time::Duration;

use continuum_shm::{
    Arena, ArenaConfig, Error, Overwritten, PAGE_ALIGN, ReadFrom, Reclaimed, RingSpec, SlabClass, SlotHandle, WriteSlot,
};

/// A uniquely named arena that's unlinked when the test ends.
struct Named {
    arena: Arena,
}

impl Drop for Named {
    fn drop(&mut self) {
        Arena::unlink(self.arena.name()).ok();
    }
}

fn arena(config: &ArenaConfig) -> Named {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let name = format!("/cnt.t.{}.{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed));
    Named { arena: Arena::create(&name, config).unwrap() }
}

fn small() -> ArenaConfig {
    ArenaConfig {
        classes: vec![SlabClass { slot_size: 100, slot_count: 2 }, SlabClass { slot_size: 40_000, slot_count: 3 }],
        rings: vec![RingSpec { stride: 8, capacity: 4 }],
        ..ArenaConfig::default()
    }
}

fn write(slot: &mut WriteSlot, bytes: &[u8]) {
    slot.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
}

fn publish(arena: &Arena, class: u16, bytes: &[u8]) -> SlotHandle {
    let mut slot = arena.claim(class, 7).unwrap();
    write(&mut slot, bytes);
    slot.publish(bytes.len()).unwrap()
}

// Wide enough that a loaded machine finishes the steps before SHORT runs out.
const SHORT: Duration = Duration::from_millis(200);
const PAST_SHORT: Duration = Duration::from_millis(400);
const LONG: Duration = Duration::from_secs(60);

#[test]
fn every_slot_address_and_size_is_16k_aligned() {
    let a = arena(&small());
    let a = &a.arena;
    assert_eq!(a.base_ptr() as usize % PAGE_ALIGN, 0);
    assert_eq!(a.len() % PAGE_ALIGN, 0);
    for class in 0..a.class_count() as u16 {
        let c = a.class(class).unwrap();
        assert_eq!(c.slot_size % PAGE_ALIGN, 0);
        let slots: Vec<_> = (0..c.slot_count).map(|_| a.claim(class, 0).unwrap()).collect();
        for mut s in slots {
            assert_eq!(s.offset() % PAGE_ALIGN, 0);
            assert_eq!(s.as_mut_slice().as_ptr() as usize % PAGE_ALIGN, 0);
        }
    }
}

#[test]
fn a_second_mapping_reads_what_the_first_published() {
    let a = arena(&small());
    let h = publish(&a.arena, 1, b"hello");
    let other = Arena::open(a.arena.name()).unwrap();
    let lease = other.lease(h, LONG).unwrap();
    let mut out = [0u8; 5];
    lease.copy_to(&mut out).unwrap();
    assert_eq!(&out, b"hello");
    assert_eq!(lease.as_ptr() as usize % PAGE_ALIGN, 0);
}

#[test]
fn an_arena_maps_from_a_file_descriptor_and_sees_the_same_bytes() {
    let a = arena(&small());
    let h = publish(&a.arena, 1, b"via fd");
    let cname = std::ffi::CString::new(a.arena.name()).unwrap();
    // SAFETY: a valid name; the fd is closed by OwnedFd.
    let raw = unsafe { libc::shm_open(cname.as_ptr(), libc::O_RDWR, 0) };
    assert!(raw >= 0);
    // SAFETY: shm_open returned a fresh fd we own.
    let fd = unsafe { std::os::fd::OwnedFd::from_raw_fd(raw) };
    let other = Arena::from_fd(fd.as_fd(), "label").unwrap();
    drop(fd); // the mapping keeps its own reference
    assert_eq!((other.name(), other.len()), ("label", a.arena.len()));
    let lease = other.lease(h, LONG).unwrap();
    let mut out = [0u8; 6];
    lease.copy_to(&mut out).unwrap();
    assert_eq!(&out, b"via fd");
}

#[test]
fn an_fd_that_is_not_an_arena_is_refused() {
    let f = std::fs::File::open("/dev/null").unwrap();
    assert!(matches!(Arena::from_fd(f.as_fd(), "null"), Err(Error::BadArena)));
}

#[test]
fn opening_checks_names_and_existence() {
    assert!(matches!(Arena::open("/cnt.no.such.arena"), Err(Error::Io(_))));
    assert!(matches!(Arena::open("no-slash"), Err(Error::BadName(_))));
    let long = format!("/{}", "x".repeat(31));
    assert!(matches!(Arena::create(&long, &small()), Err(Error::BadName(_))));
}

#[test]
fn claim_publish_lease_and_release() {
    let a = arena(&small());
    let a = &a.arena;
    let h = publish(a, 0, b"abc");
    assert_eq!(a.owner(h).unwrap(), 7);
    assert_eq!(a.pin_count(h).unwrap(), 0);
    let lease = a.lease(h, LONG).unwrap();
    let second = a.lease(h, LONG).unwrap();
    assert_eq!(a.pin_count(h).unwrap(), 2);
    assert_eq!(lease.len(), 3);
    // SAFETY: the lease is live for the whole test.
    assert_eq!(unsafe { lease.as_slice() }, b"abc");
    assert!(lease.release());
    assert!(!lease.release(), "a second release is a no-op");
    drop(second);
    assert_eq!(a.pin_count(h).unwrap(), 0);
}

#[test]
fn a_full_class_frees_up_after_a_reclaim() {
    let a = arena(&small());
    let a = &a.arena;
    let first = publish(a, 0, b"1");
    publish(a, 0, b"2");
    assert!(matches!(a.claim(0, 0), Err(Error::Full)));
    assert_eq!(a.reclaim(first).unwrap(), Reclaimed::Freed);
    let again = a.claim(0, 0).unwrap();
    assert_eq!(again.handle().index, first.index);
    assert_eq!(again.handle().generation, first.generation + 1);
}

#[test]
fn a_recycled_slot_rejects_its_old_handle_and_lease() {
    let a = arena(&small());
    let a = &a.arena;
    let old = publish(a, 0, b"old");
    let stale_lease = a.lease(old, SHORT).unwrap();
    thread::sleep(PAST_SHORT);
    assert_eq!(a.reclaim(old).unwrap(), Reclaimed::Evicted { pins: 1 });
    let new = publish(a, 0, b"new");
    let new = if new.index == old.index { new } else { publish(a, 0, b"new") };
    assert_eq!(new.index, old.index);
    assert_ne!(new.generation, old.generation);

    assert!(matches!(a.lease(old, LONG), Err(Error::Stale)));
    assert!(matches!(a.reclaim(old), Err(Error::Stale)));
    assert!(!stale_lease.is_valid());
    let live = a.lease(new, LONG).unwrap();
    assert!(!stale_lease.release(), "the evicted lease has nothing to release");
    assert_eq!(a.pin_count(new).unwrap(), 1, "and it doesn't touch the new generation's pins");
    assert!(live.is_valid());
}

#[test]
fn an_unpublished_write_frees_its_slot_with_a_new_generation() {
    let a = arena(&small());
    let a = &a.arena;
    let dropped = a.claim(0, 0).unwrap().handle();
    let other = a.claim(0, 0).unwrap();
    let reused = a.claim(0, 0).unwrap();
    assert_eq!(reused.handle().index, dropped.index);
    assert_eq!(reused.handle().generation, dropped.generation + 1);
    drop(other);
}

#[test]
fn a_live_pin_holds_a_reclaimed_slot_until_released() {
    let a = arena(&small());
    let a = &a.arena;
    let h = publish(a, 0, b"x");
    publish(a, 0, b"y");
    let lease = a.lease(h, LONG).unwrap();
    assert_eq!(a.reclaim(h).unwrap(), Reclaimed::Pinned);
    assert!(matches!(a.lease(h, LONG), Err(Error::Stale)), "no new pins once reclaiming");
    assert!(matches!(a.claim(0, 0), Err(Error::Full)));
    assert!(lease.is_valid());
    assert!(lease.release());
    assert_eq!(a.claim(0, 0).unwrap().handle().index, h.index);
}

#[test]
fn an_expired_lease_cannot_block_the_producer() {
    let a = arena(&small());
    let a = &a.arena;
    let h = publish(a, 0, b"x");
    publish(a, 0, b"y");
    let stalled = a.lease(h, SHORT).unwrap();
    assert_eq!(a.reclaim(h).unwrap(), Reclaimed::Pinned);
    assert!(matches!(a.claim(0, 0), Err(Error::Full)));
    thread::sleep(PAST_SHORT);
    let reused = a.claim(0, 0).unwrap();
    assert_eq!(reused.handle().index, h.index);
    assert!(!stalled.is_valid());
    assert!(matches!(stalled.renew(LONG), Err(Error::Expired)));
}

#[test]
fn renewing_keeps_a_standing_lease_alive() {
    let a = arena(&small());
    let a = &a.arena;
    let h = publish(a, 0, b"x");
    let lease = a.lease(h, SHORT).unwrap();
    lease.renew(LONG).unwrap();
    thread::sleep(PAST_SHORT);
    assert_eq!(a.reclaim(h).unwrap(), Reclaimed::Pinned);
    assert!(lease.is_valid());
}

#[test]
fn with_flush_required_an_expired_pin_waits_for_the_tier_1_copy() {
    let config = ArenaConfig { reclaim_requires_flush: true, ..small() };
    let a = arena(&config);
    let a = &a.arena;
    let h = publish(a, 0, b"x");
    let _stalled = a.lease(h, SHORT).unwrap();
    thread::sleep(PAST_SHORT);
    assert_eq!(a.reclaim(h).unwrap(), Reclaimed::Pinned);
    a.mark_flushed(h).unwrap();
    publish(a, 0, b"y");
    assert_eq!(a.claim(0, 0).unwrap().handle().index, h.index);
}

#[test]
fn overwriting_claims_reuse_the_oldest_unpinned_slot() {
    let a = arena(&small());
    let a = &a.arena;
    let oldest = publish(a, 1, b"0");
    let middle = publish(a, 1, b"1");
    publish(a, 1, b"2");
    let _pin = a.lease(oldest, LONG).unwrap();
    let (slot, gap) = a.claim_overwriting(1, 9).unwrap();
    assert_eq!(slot.handle().index, middle.index, "the oldest is pinned, so the next oldest goes");
    assert_eq!(gap, vec![Overwritten { handle: middle, owner: 7 }], "it wasn't flushed, so that's a Tier 1 gap");
    assert!(matches!(a.lease(middle, LONG), Err(Error::Stale)));
    let h = slot.publish(1).unwrap();
    assert_eq!(a.owner(h).unwrap(), 9);
}

#[test]
fn a_flushed_slot_is_overwritten_without_a_gap() {
    let config = ArenaConfig { reclaim_requires_flush: true, ..small() };
    let a = arena(&config);
    let a = &a.arena;
    let first = publish(a, 0, b"0");
    publish(a, 0, b"1");
    a.mark_flushed(first).unwrap();
    let (slot, gap) = a.claim_overwriting(0, 0).unwrap();
    assert_eq!((slot.handle().index, gap), (first.index, vec![]));
}

#[test]
fn a_late_flush_mark_does_not_carry_over_to_the_reused_slot() {
    let a = arena(&small());
    let a = &a.arena;
    let old = publish(a, 0, b"0");
    let other = publish(a, 0, b"1");
    assert_eq!(a.reclaim(old).unwrap(), Reclaimed::Freed);
    let reused = publish(a, 0, b"2");
    assert_eq!(reused.index, old.index);
    assert!(matches!(a.mark_flushed(old), Err(Error::Stale)), "the tier worker flushed the old generation");
    let _keep = a.lease(other, LONG).unwrap(); // so the overwrite has to pick the reused slot
    let (_, gap) = a.claim_overwriting(0, 0).unwrap();
    assert_eq!(
        gap.iter().map(|g| g.handle).collect::<Vec<_>>(),
        [reused],
        "the new generation was never flushed, so its loss is a gap"
    );
}

#[test]
fn recycling_reuses_the_oldest_flushed_slot_and_never_an_unflushed_one() {
    let config = ArenaConfig { reclaim_requires_flush: true, ..small() };
    let a = arena(&config);
    let a = &a.arena;
    let first = publish(a, 0, b"0");
    let second = publish(a, 0, b"1");
    assert!(matches!(a.claim(0, 0), Err(Error::Full)), "both Ready: a plain claim finds nothing free");
    assert!(matches!(a.claim_recycling(0, 0), Err(Error::Full)), "neither is flushed, so neither may go");
    a.mark_flushed(second).unwrap();
    let slot = a.claim_recycling(0, 9).unwrap();
    assert_eq!(slot.handle().index, second.index, "the flushed one goes, though the unflushed one is older");
    assert!(matches!(a.lease(second, LONG), Err(Error::Stale)));
    assert!(a.lease(first, LONG).is_ok(), "the unflushed slot is kept");
}

#[test]
fn overwriting_never_waits_for_a_flush() {
    let config = ArenaConfig { reclaim_requires_flush: true, ..small() };
    let a = arena(&config);
    let a = &a.arena;
    let first = publish(a, 0, b"0");
    publish(a, 0, b"1");
    let (_, gap) = a.claim_overwriting(0, 0).unwrap();
    assert_eq!(gap.iter().map(|g| g.handle).collect::<Vec<_>>(), [first]);
}

#[test]
fn rings_wrap_and_report_what_a_slow_reader_missed() {
    let a = arena(&small());
    let a = &a.arena;
    let mut writer = a.ring_writer(0).unwrap();
    assert!(matches!(a.ring_writer(0), Err(Error::RingBusy(0))));
    let mut latest = a.ring_reader(0, ReadFrom::Latest).unwrap();
    for n in 0..10u64 {
        assert_eq!(writer.push(&n.to_le_bytes()).unwrap(), n);
    }
    let mut buf = [0u8; 8];
    let first = latest.read(&mut buf).unwrap().unwrap();
    assert_eq!((first.seq, first.lost), (6, 6), "capacity 4: records 0 to 5 were overwritten");
    assert_eq!(u64::from_le_bytes(buf), 6);
    for n in 7..10u64 {
        let r = latest.read(&mut buf).unwrap().unwrap();
        assert_eq!((r.seq, r.lost, u64::from_le_bytes(buf)), (n, 0, n));
    }
    assert_eq!(latest.read(&mut buf).unwrap(), None);

    let mut oldest = a.ring_reader(0, ReadFrom::Oldest).unwrap();
    assert_eq!(oldest.read(&mut buf).unwrap().unwrap().seq, 6);
    assert!(matches!(writer.push(&[0; 3]), Err(Error::BadRecord { got: 3, stride: 8 })));
    drop(writer);
    a.ring_writer(0).unwrap();
}

/// #3210: a timeout past what an `Instant` holds (`Duration::MAX`, the usual "forever") panicked before the wait.
#[test]
fn a_ring_wait_without_end_answers_a_published_record() {
    let a = arena(&small());
    let a = a.arena.clone();
    let mut writer = a.ring_writer(0).unwrap();
    let reader = a.ring_reader(0, ReadFrom::Latest).unwrap();
    writer.push(&[1; 8]).unwrap();
    assert!(reader.wait(Duration::MAX), "a record is ready");
    Arena::unlink(a.name()).ok();
}

#[test]
fn a_blocked_ring_reader_wakes_on_the_next_record() {
    let a = arena(&small());
    let a = a.arena.clone();
    let mut writer = a.ring_writer(0).unwrap();
    writer.push(&[0; 8]).unwrap();
    let reader = a.ring_reader(0, ReadFrom::Latest).unwrap();
    assert!(!reader.wait(Duration::from_millis(10)), "nothing new yet");
    let waiting = thread::spawn(move || {
        let woke = reader.wait(Duration::from_secs(2));
        (woke, std::time::Instant::now())
    });
    thread::sleep(Duration::from_millis(20)); // let the reader block
    let pushed = std::time::Instant::now();
    writer.push(&[1; 8]).unwrap();
    let (woke, at) = waiting.join().unwrap();
    assert!(woke, "missed the wake and timed out");
    // The target is under 25 ms; the bound is looser so a loaded machine doesn't fail it, and a missed wake would
    // take the full 2 s timeout.
    assert!(at - pushed < Duration::from_millis(250), "woke {:?} after the push", at - pushed);
    Arena::unlink(a.name()).ok();
}

/// Record `n`: the bytes of `n` (little-endian u64), repeated to fill `stride`.
fn stamped(n: u64, stride: usize) -> Vec<u8> {
    n.to_le_bytes().iter().copied().cycle().take(stride).collect()
}

/// A writer pushes 50,000 stamped records while two readers check every record they get is whole.
fn readers_never_see_a_torn_record(stride: u32) {
    let config = ArenaConfig { rings: vec![RingSpec { stride, capacity: 8 }], ..ArenaConfig::default() };
    let a = arena(&config);
    let a = a.arena.clone();
    let mut writer = a.ring_writer(0).unwrap();
    let done = Arc::new(AtomicBool::new(false));
    let readers: Vec<_> = (0..2)
        .map(|_| {
            let mut reader = a.ring_reader(0, ReadFrom::Oldest).unwrap();
            let done = done.clone();
            thread::spawn(move || {
                let mut buf = vec![0u8; stride as usize];
                let mut read = 0u64;
                loop {
                    match reader.read(&mut buf).unwrap() {
                        Some(r) => {
                            assert_eq!(buf, stamped(r.seq, stride as usize), "torn record {}", r.seq);
                            read += 1;
                        }
                        None if done.load(Ordering::SeqCst) => return read,
                        None => thread::yield_now(),
                    }
                }
            })
        })
        .collect();
    for n in 0..50_000u64 {
        writer.push(&stamped(n, stride as usize)).unwrap();
    }
    done.store(true, Ordering::SeqCst);
    for r in readers {
        assert!(r.join().unwrap() > 0);
    }
    Arena::unlink(a.name()).ok();
}

#[test]
fn concurrent_ring_readers_never_see_a_torn_record() {
    readers_never_see_a_torn_record(64); // copied in 8-byte words
}

#[test]
fn concurrent_ring_readers_never_see_a_torn_record_with_an_odd_stride() {
    readers_never_see_a_torn_record(12); // not a multiple of 8, so copied byte by byte
}

#[test]
fn a_renewed_lease_on_a_reclaiming_slot_holds_it_until_released() {
    let a = arena(&small());
    let a = &a.arena;
    let h = publish(a, 0, b"x");
    publish(a, 0, b"y");
    let lease = a.lease(h, SHORT).unwrap();
    assert_eq!(a.reclaim(h).unwrap(), Reclaimed::Pinned);
    lease.renew(LONG).unwrap();
    thread::sleep(PAST_SHORT);
    assert!(matches!(a.claim(0, 0), Err(Error::Full)), "the renewed pin is live, so the slot isn't evicted");
    assert!(lease.is_valid());
    assert!(lease.release());
    assert_eq!(a.claim(0, 0).unwrap().handle().index, h.index, "the last release frees it");
}

#[test]
fn concurrent_leases_always_see_the_bytes_they_pinned() {
    let config = ArenaConfig { classes: vec![SlabClass { slot_size: 64, slot_count: 4 }], ..ArenaConfig::default() };
    let a = arena(&config);
    let a = a.arena.clone();
    let latest = Arc::new(std::sync::Mutex::new(None::<SlotHandle>));
    let done = Arc::new(AtomicBool::new(false));
    let readers: Vec<_> = (0..3)
        .map(|_| {
            let (a, latest, done) = (a.clone(), latest.clone(), done.clone());
            thread::spawn(move || {
                let mut pinned = 0;
                while !done.load(Ordering::SeqCst) {
                    let Some(h) = *latest.lock().unwrap() else { continue };
                    let Ok(lease) = a.lease(h, LONG) else { continue };
                    // SAFETY: the lease is long and released below.
                    let bytes = unsafe { lease.as_slice() };
                    let stamp = bytes[0];
                    assert!(bytes.iter().all(|&b| b == stamp), "a pinned slot changed under its lease");
                    pinned += 1;
                }
                pinned
            })
        })
        .collect();
    for n in 0..5_000u32 {
        let mut slot = loop {
            match a.claim_overwriting(0, 0) {
                Ok((s, _)) => break s,
                Err(Error::Full) => thread::yield_now(),
                Err(e) => panic!("{e}"),
            }
        };
        slot.as_mut_slice()[..64].fill(n as u8);
        *latest.lock().unwrap() = Some(slot.publish(64).unwrap());
    }
    done.store(true, Ordering::SeqCst);
    for r in readers {
        r.join().unwrap();
    }
    Arena::unlink(a.name()).ok();
}

#[test]
fn scan_reports_each_slot_and_purge_zeroes_one_and_stales_its_readers() {
    let a = arena(&small());
    let mut slot = a.arena.claim(0, 1).unwrap();
    slot.as_mut_slice()[..4].copy_from_slice(&[9, 9, 9, 9]);
    let h = slot.publish(4).unwrap();
    let lease = a.arena.lease(h, Duration::from_secs(5)).unwrap();
    let scan = a.arena.scan(0).unwrap();
    let me = scan.iter().find(|s| s.handle == h).unwrap();
    assert_eq!((me.state, me.pins, me.flushed), (continuum_shm::SlotState::Ready, 1, false));
    assert!(scan.iter().filter(|s| s.handle != h).all(|s| s.state == continuum_shm::SlotState::Free));

    // A purge takes it out of use at once, but leaves its bytes alone while a live lease holds them.
    assert_eq!(a.arena.purge(h).unwrap(), continuum_shm::Purged::Pending);
    assert!(lease.is_valid(), "the reader's lease still holds");
    // SAFETY: the slot lies inside the mapping, and nothing writes it now.
    let held = unsafe { std::slice::from_raw_parts(a.arena.base_ptr().add(lease.offset()), 4) };
    assert_eq!(held, [9, 9, 9, 9], "a live lease's bytes don't change");
    assert!(matches!(a.arena.lease(h, Duration::from_secs(5)), Err(Error::Stale)), "no new leases");
    assert!(matches!(a.arena.claim(0, 2).map(|_| ()), Ok(())), "other slots still claim");
    // Once the reader lets go, the purge zeroes it and frees it at a new generation.
    drop(lease);
    assert_eq!(a.arena.purge(h).unwrap(), continuum_shm::Purged::Zeroed);
    let after = a.arena.scan(0).unwrap();
    let slot_now = after.iter().find(|s| s.handle.index == h.index).unwrap();
    assert_eq!(slot_now.state, continuum_shm::SlotState::Free);
    assert_ne!(slot_now.handle.generation, h.generation);
    let off = a.arena.slot_offset(h.class, h.index).unwrap();
    // SAFETY: the slot lies inside the mapping; nothing writes it now.
    let bytes = unsafe { std::slice::from_raw_parts(a.arena.base_ptr().add(off), 4) };
    assert_eq!(bytes, [0, 0, 0, 0]);
    assert!(matches!(a.arena.purge(h), Err(Error::Stale)), "nothing left to purge");
}

#[test]
fn a_reader_seeks_to_a_record_and_a_lapped_one_counts_as_lost() {
    let a = arena(&small());
    let a = &a.arena;
    let mut writer = a.ring_writer(0).unwrap();
    for n in 0..6u64 {
        writer.push(&n.to_le_bytes()).unwrap();
    }
    let mut reader = a.ring_reader(0, ReadFrom::Oldest).unwrap();
    let mut out = [0u8; 8];
    // The ring holds 4: records 2 to 5.
    reader.seek(3);
    assert_eq!(reader.position(), 3);
    let r = reader.read(&mut out).unwrap().unwrap();
    assert_eq!((r.seq, r.lost, u64::from_le_bytes(out)), (3, 0, 3));
    // Record 0 was overwritten: the reader lands on the oldest left, and says how many it skipped.
    reader.seek(0);
    let r = reader.read(&mut out).unwrap().unwrap();
    assert_eq!((r.seq, r.lost, u64::from_le_bytes(out)), (2, 2, 2));
}

#[test]
fn used_counts_the_slots_that_arent_free() {
    let named = arena(&ArenaConfig { classes: vec![SlabClass { slot_size: 4096, slot_count: 4 }], ..small() });
    let a = &named.arena;
    assert_eq!(a.used(0).unwrap(), 0);
    let writing = a.claim(0, 1).unwrap();
    a.claim(0, 1).unwrap().publish(10).unwrap();
    assert_eq!(a.used(0).unwrap(), 2, "one being written, one ready");
    drop(writing); // an unpublished slot goes back to free
    assert_eq!(a.used(0).unwrap(), 1);
    assert!(a.used(9).is_err());
}

#[test]
fn held_counts_only_what_cant_be_taken_back() {
    let named = arena(&ArenaConfig { classes: vec![SlabClass { slot_size: 4096, slot_count: 4 }], ..small() });
    let a = &named.arena;
    let writing = a.claim(0, 1).unwrap();
    let unflushed = a.claim(0, 1).unwrap().publish(10).unwrap();
    let flushed = a.claim(0, 1).unwrap().publish(10).unwrap();
    a.mark_flushed(flushed).unwrap();
    // In use: three. Held: the one being written and the unflushed one; the flushed one could be overwritten freely.
    assert_eq!((a.used(0).unwrap(), a.held(0).unwrap()), (3, 2));
    // A live pin holds a flushed slot too.
    let lease = a.lease(flushed, std::time::Duration::from_secs(5)).unwrap();
    assert_eq!(a.held(0).unwrap(), 3);
    drop(lease);
    drop(writing);
    a.mark_flushed(unflushed).unwrap();
    assert_eq!(a.held(0).unwrap(), 0);
    assert!(a.held(9).is_err());
}
