//! Hotplug from a device list that is read again and again (macOS: AVFoundation's cameras and CoreAudio's input
//! devices): each read is compared with the last, and what appeared or went becomes the supervisor's events. Device
//! lists are cheap to read, and a short poll is simpler and easier to trust than notifications that need a run loop
//! of their own; the supervisor's settle delay (1.5 s) is longer than the poll anyway.

use std::collections::BTreeMap;

use crate::supervisor::{DeviceInfo, Event};

/// How often a device list is read again.
pub const POLL_MS: u64 = 1_000;

/// The devices seen at the last read, by key.
#[derive(Clone, Debug, Default)]
pub struct DeviceList {
    seen: BTreeMap<String, DeviceInfo>,
}

impl DeviceList {
    /// An empty list: the first read reports every device as added.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Compares a fresh read with the last: `Removed` for each device that went, then `Added` for each that
    /// appeared, in key order. A device whose description changed under the same key counts as gone and back.
    pub fn update(&mut self, now: Vec<DeviceInfo>) -> Vec<Event> {
        let now: BTreeMap<String, DeviceInfo> = now.into_iter().map(|d| (d.key.clone(), d)).collect();
        let mut out = Vec::new();
        for (key, old) in &self.seen {
            if now.get(key) != Some(old) {
                out.push(Event::Removed(key.clone()));
            }
        }
        for (key, info) in &now {
            if self.seen.get(key) != Some(info) {
                out.push(Event::Added(info.clone()));
            }
        }
        self.seen = now;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::DeviceKind;

    fn dev(key: &str, name: &str) -> DeviceInfo {
        DeviceInfo {
            key: key.into(),
            kind: DeviceKind::Microphone,
            name: name.into(),
            usb: None,
            node: key.into(),
            paths: vec![],
        }
    }

    #[test]
    fn reads_become_added_and_removed_events() {
        let mut l = DeviceList::new();
        assert_eq!(
            l.update(vec![dev("a", "Mic A"), dev("b", "Mic B")]),
            vec![Event::Added(dev("a", "Mic A")), Event::Added(dev("b", "Mic B"))]
        );
        assert!(l.update(vec![dev("b", "Mic B"), dev("a", "Mic A")]).is_empty(), "the same devices, in any order");
        assert_eq!(l.update(vec![dev("b", "Mic B")]), vec![Event::Removed("a".into())]);
        assert_eq!(l.update(vec![dev("b", "Mic B"), dev("a", "Mic A")]), vec![Event::Added(dev("a", "Mic A"))]);
        // Same key, new description: gone and back.
        assert_eq!(
            l.update(vec![dev("b", "Mic B2"), dev("a", "Mic A")]),
            vec![Event::Removed("b".into()), Event::Added(dev("b", "Mic B2"))]
        );
        assert_eq!(l.update(vec![]).len(), 2);
    }
}
