//! Machine profiles: the memory and compute budgets the scheduler admits leases against (docs/CONTINUUM-BROKER.md,
//! "Scheduler").
//!
//! Running the Linux rig under [`Profile::mac_mini_64gb`] enforces the Mac's limits there, which is the point of
//! profiles: nothing built on two 3090s should surprise the 64 GB Mac. Every number here is a setting with a first
//! guess as its default, to be tuned from measured use.

use std::collections::BTreeMap;

use continuum_core::ComputeDomain;

use crate::cost::CostWeights;

/// One shared memory size class as the scheduler sees it: how big a slot is and how many there are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotClass {
    /// A slot's size in bytes, a multiple of `continuum_core::PAGE_ALIGN`.
    pub size_bytes: u64,
    /// How many slots the class has.
    pub count: u32,
}

/// The budgets for one machine.
#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    /// The profile's name, as the setting names it (`mac_mini_64gb`, `linux_dual_3090`).
    pub name: String,
    /// RAM that worker leases may hold in total, in MB. Leaves room for the OS, the broker, and the Tier 0 arena.
    pub ram_mb: u64,
    /// VRAM per discrete GPU device, in MB, in device order. Empty on a machine whose GPU shares `ram_mb` (the Mac).
    pub vram_mb: Vec<u64>,
    /// The size classes of the Tier 0 arena, in class order. Capture rings are not here: they are leased at
    /// registration and never compete with jobs.
    pub slot_classes: Vec<SlotClass>,
    /// How many jobs may run at once on each compute domain. A domain missing here (the Neural Engine on Linux)
    /// has no capacity, and a capability needing it is never granted.
    pub domain_slots: BTreeMap<ComputeDomain, u32>,
    /// How compute time and memory turn into one cost number for comparing plans.
    pub weights: CostWeights,
}

impl Profile {
    /// The field rig: an Apple Silicon Mac with 64 GB unified memory. First guesses: 40 GB for worker leases, a
    /// Tier 0 arena of about 5 GB in size classes, and two jobs at a time on each accelerator.
    #[must_use]
    pub fn mac_mini_64gb() -> Self {
        Self {
            name: "mac_mini_64gb".to_owned(),
            ram_mb: 40_000,
            vram_mb: Vec::new(),
            slot_classes: vec![
                SlotClass { size_bytes: 64 * 1024, count: 2048 },
                SlotClass { size_bytes: 2 * 1024 * 1024, count: 512 },
                SlotClass { size_bytes: 16 * 1024 * 1024, count: 256 },
            ],
            domain_slots: BTreeMap::from([
                (ComputeDomain::CPU, 4),
                (ComputeDomain::GPU, 2),
                (ComputeDomain::ANE, 2),
                (ComputeDomain::MediaEngine, 2),
            ]),
            weights: CostWeights::default(),
        }
    }

    /// The lab rig: Linux, 128 GB RAM, two RTX 3090s with 24 GB each. First guesses: 96 GB for worker leases, the
    /// 10.25 GB arena from the design doc, four GPU jobs at a time (two per device), and no Neural Engine.
    #[must_use]
    pub fn linux_dual_3090() -> Self {
        Self {
            name: "linux_dual_3090".to_owned(),
            ram_mb: 96_000,
            vram_mb: vec![24_000, 24_000],
            slot_classes: vec![
                SlotClass { size_bytes: 64 * 1024, count: 4096 },
                SlotClass { size_bytes: 2 * 1024 * 1024, count: 1024 },
                SlotClass { size_bytes: 16 * 1024 * 1024, count: 512 },
            ],
            domain_slots: BTreeMap::from([
                (ComputeDomain::CPU, 8),
                (ComputeDomain::GPU, 4),
                (ComputeDomain::MediaEngine, 2),
            ]),
            weights: CostWeights::default(),
        }
    }

    /// The profile with this name, or `None` for a name this crate doesn't know.
    #[must_use]
    pub fn named(name: &str) -> Option<Self> {
        match name {
            "mac_mini_64gb" => Some(Self::mac_mini_64gb()),
            "linux_dual_3090" => Some(Self::linux_dual_3090()),
            _ => None,
        }
    }

    /// The whole arena's slot bytes across size classes.
    #[must_use]
    pub fn arena_bytes(&self) -> u64 {
        self.slot_classes.iter().map(|c| c.size_bytes * u64::from(c.count)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linux_arena_matches_the_design_doc() {
        // 256 MB + 2 GB + 8 GB, 10.25 GB in all.
        let gib = 1024 * 1024 * 1024;
        assert_eq!(Profile::linux_dual_3090().arena_bytes(), 10 * gib + gib / 4);
    }

    #[test]
    fn profiles_are_found_by_their_setting_name() {
        assert_eq!(Profile::named("mac_mini_64gb").map(|p| p.vram_mb.len()), Some(0));
        assert_eq!(Profile::named("linux_dual_3090").map(|p| p.vram_mb.len()), Some(2));
        assert!(Profile::named("raspberry_pi").is_none());
    }

    #[test]
    fn linux_has_no_neural_engine() {
        assert!(!Profile::linux_dual_3090().domain_slots.contains_key(&ComputeDomain::ANE));
        assert_eq!(Profile::mac_mini_64gb().domain_slots.get(&ComputeDomain::ANE), Some(&2));
    }
}
