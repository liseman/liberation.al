//! Read-only census of a Tier 0 arena: per class, how many slots are free, being written, ready (flushed or not),
//! reclaiming, or pinned, and per owner tag how many it holds and whether that owner is a live process.
//! Read-only diagnostic: it only loads atomics, never claims, publishes, or reclaims.
//!
//!   cargo run --release -p continuum-shm --example tier0_census -- /cnt.<id>.<generation>

use std::collections::BTreeMap;

use continuum_shm::{Arena, SlotState};

fn main() {
    let name = std::env::args().nth(1).expect("usage: tier0_census <arena name>");
    let arena = Arena::open(&name).expect("open the arena");
    for class in 0..arena.class_count() as u16 {
        let info = arena.class(class).expect("class");
        let scan = arena.scan(class).expect("scan");
        let mut states = [0u32; 4];
        let (mut flushed, mut pinned) = (0u32, 0u32);
        let mut owners: BTreeMap<u64, (u32, u32, u32)> = BTreeMap::new(); // ready, of which flushed, writing
        for s in &scan {
            let i = match s.state {
                SlotState::Free => 0,
                SlotState::Writing => 1,
                SlotState::Ready => 2,
                _ => 3,
            };
            states[i] += 1;
            flushed += u32::from(s.state == SlotState::Ready && s.flushed);
            pinned += u32::from(s.pins > 0);
            if matches!(s.state, SlotState::Ready | SlotState::Writing)
                && let Ok(owner) = arena.owner(s.handle)
            {
                let e = owners.entry(owner).or_default();
                if s.state == SlotState::Ready {
                    e.0 += 1;
                    e.1 += u32::from(s.flushed);
                } else {
                    e.2 += 1;
                }
            }
        }
        println!(
            "class {class} ({} KiB x {}): free {} writing {} ready {} (flushed {flushed}) reclaiming {} pinned {pinned}",
            info.slot_size / 1024,
            info.slot_count,
            states[0],
            states[1],
            states[2],
            states[3]
        );
        for (owner, (ready, done, writing)) in owners {
            let alive = owner <= u64::from(u32::MAX) && std::path::Path::new(&format!("/proc/{owner}")).exists();
            let comm = std::fs::read_to_string(format!("/proc/{owner}/comm")).unwrap_or_default();
            println!(
                "  owner {owner} ({}{}): ready {ready} (flushed {done}) writing {writing}",
                if alive { "alive" } else { "gone" },
                if comm.is_empty() { String::new() } else { format!(", {}", comm.trim()) }
            );
        }
    }
}
