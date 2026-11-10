//! Rings carved from an arena's reserved ring region while it's live: layout checks, generations, and the same
//! reader and writer as the superblock's rings.

use std::sync::atomic::{AtomicU32, Ordering};

use continuum_shm::{Arena, ArenaConfig, CarvedRing, Error, PAGE_ALIGN, ReadFrom, SlabClass};

struct Named {
    arena: Arena,
}

impl Drop for Named {
    fn drop(&mut self) {
        Arena::unlink(self.arena.name()).ok();
    }
}

fn arena(region: usize) -> Named {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let name = format!("/cnt.c.{}.{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed));
    let config = ArenaConfig {
        classes: vec![SlabClass { slot_size: 1, slot_count: 1 }],
        ring_region_bytes: region,
        ..ArenaConfig::default()
    };
    Named { arena: Arena::create(&name, &config).unwrap() }
}

const PAGE: u64 = PAGE_ALIGN as u64;

#[test]
fn the_region_is_page_aligned_at_the_end_and_another_mapping_sees_it() {
    let a = arena(PAGE_ALIGN * 3 + 1);
    let (start, len) = a.arena.ring_region();
    assert_eq!((start % PAGE, len), (0, 4 * PAGE), "rounded up to whole pages");
    assert_eq!(start + len, a.arena.len() as u64, "at the end of the arena");
    assert_eq!(Arena::open(a.arena.name()).unwrap().ring_region(), (start, len));
    assert_eq!(arena(0).arena.ring_region().1, 0);
}

#[test]
fn a_carved_ring_carries_records_between_mappings() {
    let a = arena(8 * PAGE_ALIGN);
    let (start, _) = a.arena.ring_region();
    // A 1,440 byte PCM block (10 ms at 48 kHz, mono 24-bit), which no preset class would fit exactly.
    let ring = a.arena.carve_ring(start, 1_440, 50).unwrap();
    assert_eq!(ring.generation, 1);
    assert_eq!(ring.data_offset, start + PAGE);
    let other = Arena::open(a.arena.name()).unwrap();
    let mut w = other.carved_ring_writer(&ring).unwrap();
    let mut r = a.arena.carved_ring_reader(&ring, ReadFrom::Oldest).unwrap();
    for n in 0..120u8 {
        w.push(&[n; 1_440]).unwrap();
    }
    assert_eq!(a.arena.carved_ring_head(&ring).unwrap(), 120);
    let mut out = [0u8; 1_440];
    let first = r.read(&mut out).unwrap().unwrap();
    assert_eq!((first.seq, first.lost, out[0]), (70, 70, 70), "the 50 newest are kept");
    // A second ring after the first one's span.
    let next = start + CarvedRing::span(1_440, 50).unwrap();
    let imu = a.arena.carve_ring(next, 64, 100).unwrap();
    other.carved_ring_writer(&imu).unwrap().push(&[9; 64]).unwrap();
    assert_eq!(a.arena.carved_ring_head(&ring).unwrap(), 120, "rings don't share records");
}

#[test]
fn a_block_in_use_or_a_ring_that_doesnt_fit_is_refused() {
    let a = arena(4 * PAGE_ALIGN);
    let (start, len) = a.arena.ring_region();
    a.arena.carve_ring(start, 8, 10).unwrap();
    assert!(matches!(a.arena.carve_ring(start, 8, 10), Err(Error::BadConfig(_))), "carved already");
    // Too big for what's left, not page aligned, outside the region, or empty.
    assert!(matches!(a.arena.carve_ring(start + 2 * PAGE, 1, 3 * PAGE as u32), Err(Error::NoSuchRing)));
    assert!(matches!(a.arena.carve_ring(start + 2 * PAGE + 128, 8, 10), Err(Error::NoSuchRing)));
    assert!(matches!(a.arena.carve_ring(start + len, 8, 10), Err(Error::NoSuchRing)));
    assert!(matches!(a.arena.carve_ring(0, 8, 10), Err(Error::NoSuchRing)), "the superblock isn't the region");
    assert!(matches!(a.arena.carve_ring(start + 2 * PAGE, 0, 10), Err(Error::NoSuchRing)));
}

#[test]
fn an_announced_ring_is_checked_before_it_is_used() {
    let a = arena(4 * PAGE_ALIGN);
    let (start, _) = a.arena.ring_region();
    let ring = a.arena.carve_ring(start, 8, 10).unwrap();
    let forged = |f: &dyn Fn(&mut CarvedRing)| {
        let mut r = ring;
        f(&mut r);
        r
    };
    // A wrong generation, or a block nobody carved, is stale; numbers outside the region or overlapping are refused.
    assert!(matches!(a.arena.carved_ring_reader(&forged(&|r| r.generation = 2), ReadFrom::Oldest), Err(Error::Stale)));
    let uncarved = forged(&|r| {
        r.ctrl_offset += 2 * PAGE;
        r.data_offset += 2 * PAGE;
    });
    assert!(matches!(a.arena.carved_ring_writer(&uncarved), Err(Error::Stale)));
    assert!(matches!(a.arena.carved_ring_head(&forged(&|r| r.data_offset = 0)), Err(Error::NoSuchRing)));
    assert!(matches!(a.arena.carved_ring_head(&forged(&|r| r.data_offset = r.ctrl_offset)), Err(Error::NoSuchRing)));
    assert!(matches!(a.arena.carved_ring_head(&forged(&|r| r.capacity = u32::MAX)), Err(Error::NoSuchRing)));
    assert!(matches!(a.arena.carved_ring_head(&forged(&|r| r.ctrl_offset += 8)), Err(Error::NoSuchRing)));
}

#[test]
fn one_writer_at_a_time_and_a_dead_writers_claim_can_be_released() {
    let a = arena(4 * PAGE_ALIGN);
    let (start, _) = a.arena.ring_region();
    let ring = a.arena.carve_ring(start, 8, 10).unwrap();
    let w = a.arena.carved_ring_writer(&ring).unwrap();
    assert!(matches!(a.arena.carved_ring_writer(&ring), Err(Error::RingBusy(_))));
    // A writer whose process died never drops its guard.
    std::mem::forget(w);
    assert!(matches!(a.arena.carved_ring_writer(&ring), Err(Error::RingBusy(_))));
    a.arena.release_carved_ring_writer(&ring).unwrap();
    a.arena.carved_ring_writer(&ring).unwrap().push(&[1; 8]).unwrap();
    assert_eq!(a.arena.carved_ring_head(&ring).unwrap(), 1);
}

#[test]
fn a_released_writer_stops_and_its_drop_leaves_the_next_writers_claim_alone() {
    let a = arena(4 * PAGE_ALIGN);
    let (start, _) = a.arena.ring_region();
    let ring = a.arena.carved_ring_writer(&a.arena.carve_ring(start, 8, 10).unwrap());
    let mut old = ring.unwrap();
    let carved = a.arena.carved_rings()[0];
    old.push(&[1; 8]).unwrap();
    // The broker releases it (its session ended) while the process is still running, and a new writer takes over.
    a.arena.release_carved_ring_writer(&carved).unwrap();
    let mut new = a.arena.carved_ring_writer(&carved).unwrap();
    assert!(matches!(old.push(&[2; 8]), Err(Error::NotWriter)), "the old writer can't write any more");
    drop(old);
    assert!(matches!(a.arena.carved_ring_writer(&carved), Err(Error::RingBusy(_))), "its Drop left the new claim");
    new.push(&[3; 8]).unwrap();
    assert_eq!(a.arena.carved_ring_head(&carved).unwrap(), 2, "one record each, none from after the release");
}

#[test]
fn the_superblock_records_each_carved_ring_and_readers_check_against_it() {
    let a = arena(8 * PAGE_ALIGN);
    let (start, _) = a.arena.ring_region();
    let first = a.arena.carve_ring(start, 1_440, 50).unwrap();
    let second = a.arena.carve_ring(start + CarvedRing::span(1_440, 50).unwrap(), 64, 10).unwrap();
    // Another process sees both, in carving order (the order is each ring's index).
    let other = Arena::open(a.arena.name()).unwrap();
    assert_eq!(other.carved_rings(), [first, second]);
    // An announcement that disagrees with the record (here a smaller capacity, same block) is refused.
    let wrong = CarvedRing { capacity: 10, ..first };
    assert!(matches!(other.carved_ring_reader(&wrong, ReadFrom::Oldest), Err(Error::NoSuchRing)));
    // A carve that would lie in another ring's pages is refused before it touches them.
    let inside = first.data_offset + PAGE;
    assert!(matches!(a.arena.carve_ring(inside, 8, 10), Err(Error::BadConfig(m)) if m.contains("overlap")));
    assert_eq!(a.arena.carved_rings().len(), 2);
}

#[test]
fn a_release_for_one_process_leaves_another_processs_claim_alone() {
    let a = arena(4 * PAGE_ALIGN);
    let (start, _) = a.arena.ring_region();
    let ring = a.arena.carve_ring(start, 8, 10).unwrap();
    let w = a.arena.carved_ring_writer(&ring).unwrap();
    // A late close of some other process's session: this claim isn't its to release.
    assert!(!a.arena.release_carved_ring_writer_of(&ring, std::process::id().wrapping_add(1)).unwrap());
    assert!(matches!(a.arena.carved_ring_writer(&ring), Err(Error::RingBusy(_))));
    // This process's: released.
    assert!(a.arena.release_carved_ring_writer_of(&ring, std::process::id()).unwrap());
    drop(w);
    a.arena.carved_ring_writer(&ring).unwrap();
}
