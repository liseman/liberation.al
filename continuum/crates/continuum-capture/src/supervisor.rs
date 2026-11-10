//! The capture supervisor's rules (docs/CONTINUUM.md, "One process per physical device"), as a state machine.
//!
//! It is given events (a device appeared or went, a worker said its device is open, a worker exited, time passed)
//! and answers with actions (start a worker, stop one). It holds no devices and no processes itself, so every rule is
//! tested by feeding it events: the driver that turns hotplug notices into events and actions into processes is thin.
//!
//! The rules:
//! - A device that appears and matches a manifest entry gets one worker. A device that matches nothing is ignored.
//! - One open token at a time: only one worker may be opening its device, and the next may start only once that one
//!   said it opened (or exited, or ran out of time) and the settle delay has passed. Concurrent UVC opens fail.
//! - A worker that exits is restarted after a backoff that doubles with each quick failure, up to a cap, and goes
//!   back to the first step once a run lasted long enough to count as healthy.
//! - A worker that doesn't open in time is stopped, which counts as a failed open. It keeps the open token until it
//!   exits; one that ignores the stop is killed, and one that outlives even that (a UVC ioctl stuck in the kernel)
//!   is given up on, so the token and the other devices aren't held hostage.
//! - A device that goes has its worker stopped and is not restarted, unless it comes back before the worker has
//!   exited (a USB re-enumeration): then it's started again once the old worker is gone.
//! - A manifest entry gets one worker at a time, whatever hotplug lists (a camera's metadata node, a second alias).
//! - A paused device has no worker: pausing stops its worker, and while paused it waits without opening.
//!   Resuming starts it again right away, with the open token as usual.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::manifest::{DeviceKind, DeviceMatch, Manifest, SupervisorSettings, parse_usb};

/// A device as hotplug reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Stable while it's plugged in: a sysfs path, a PipeWire node id, an IOKit registry id.
    pub key: String,
    /// What it is, as hotplug knows it.
    pub kind: DeviceKind,
    /// Its name (the V4L2 card, the audio device's description).
    pub name: String,
    /// USB vendor and product, when it's a USB device.
    pub usb: Option<(u16, u16)>,
    /// The node a worker opens (`/dev/video0`, a PipeWire node name).
    pub node: String,
    /// Stable paths or ids it can be matched by (`/dev/v4l/by-id/...`).
    pub paths: Vec<String>,
}

/// What happened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// A device appeared.
    Added(DeviceInfo),
    /// A device went, by key.
    Removed(String),
    /// A worker opened its device: the open token is free again.
    Opened(WorkerId),
    /// A worker exited (or was stopped).
    Exited(WorkerId),
}

/// What to do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    /// Start a worker for this manifest device on this hotplug device.
    Start {
        /// The id to give it; its events come back with it.
        worker: WorkerId,
        /// The manifest entry's name.
        device: String,
        /// The device to open.
        info: DeviceInfo,
    },
    /// Stop this worker (`SIGTERM`).
    Stop(WorkerId),
    /// Kill this worker: it ignored the stop (`SIGKILL`).
    Kill(WorkerId),
}

/// A worker, by the supervisor's own count: never reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkerId(pub u64);

#[derive(Clone, Debug)]
enum State {
    /// Waiting for the open token, or for its restart time.
    Waiting { not_before_ms: u64 },
    /// Started, not yet opened: it holds the token.
    Opening { worker: WorkerId, deadline_ms: u64 },
    /// Opened its device and is capturing.
    /// `opened_ms`: when it said so, which a healthy run is counted from (the open itself isn't recording).
    Running { worker: WorkerId, opened_ms: u64 },
    /// Asked to stop: its device went, or it didn't open in time.
    Stopping {
        worker: WorkerId,
        /// Start again once it has exited: with this device (it came back while this one was stopping), or with the
        /// slot's own after a backoff (a failed open). `None`: its device went for good.
        then: Then,
        /// It keeps the open token until it exits (a failed open: it may still be in the middle of opening).
        holds_token: bool,
        /// When to escalate: kill it, or, once killed, give up waiting.
        deadline_ms: u64,
        killed: bool,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Then {
    /// Its device went: forget the slot.
    Forget,
    /// It failed to open: restart after the backoff.
    Backoff,
    /// Its device came back (possibly under a new key): start on this, right away.
    Readd(DeviceInfo),
}

#[derive(Clone, Debug)]
struct Slot {
    device: String,
    info: DeviceInfo,
    state: State,
    backoff_ms: u64,
}

/// The supervisor's state.
#[derive(Clone, Debug)]
pub struct Supervisor {
    manifest: Manifest,
    settings: SupervisorSettings,
    /// By hotplug key.
    slots: BTreeMap<String, Slot>,
    /// Keys waiting for the token, in arrival order.
    queue: VecDeque<String>,
    /// The token is free from this time (after the last open's settle delay).
    token_free_ms: u64,
    next_worker: u64,
    /// Manifest entries paused for privacy: never started while here.
    paused: BTreeSet<String>,
}

impl Supervisor {
    /// A supervisor for this manifest, with no devices yet.
    #[must_use]
    pub fn new(manifest: Manifest) -> Self {
        let settings = manifest.supervisor;
        let paused = paused_in(&manifest);
        Self {
            manifest,
            settings,
            slots: BTreeMap::new(),
            queue: VecDeque::new(),
            token_free_ms: 0,
            next_worker: 1,
            paused,
        }
    }

    /// The manifest entries paused now.
    #[must_use]
    pub fn paused(&self) -> &BTreeSet<String> {
        &self.paused
    }

    /// Follows a new set of paused manifest entries: a newly paused device's worker is stopped (it then waits without
    /// opening), and a resumed one is due to start at once. Returns what to do.
    pub fn set_paused(&mut self, paused: BTreeSet<String>, now_ms: u64) -> Vec<Action> {
        let mut out = Vec::new();
        if paused == self.paused {
            return out;
        }
        let stop_grace_ms = self.settings.stop_grace_ms;
        for slot in self.slots.values_mut() {
            let now_paused = paused.contains(&slot.device);
            let was_paused = self.paused.contains(&slot.device);
            match slot.state.clone() {
                State::Opening { worker, .. } | State::Running { worker, .. } if now_paused => {
                    // An opening worker keeps the open token until it exits, as with a failed open.
                    let holds_token = matches!(slot.state, State::Opening { .. });
                    slot.state = State::Stopping {
                        worker,
                        then: Then::Readd(slot.info.clone()),
                        holds_token,
                        deadline_ms: now_ms + stop_grace_ms,
                        killed: false,
                    };
                    out.push(Action::Stop(worker));
                }
                State::Waiting { .. } if was_paused && !now_paused => {
                    slot.state = State::Waiting { not_before_ms: now_ms };
                    slot.backoff_ms = self.settings.backoff_initial_ms;
                }
                _ => {}
            }
        }
        self.paused = paused;
        self.tick(now_ms, &mut out);
        out
    }

    /// The manifest entry a device matches, if any (the first that does).
    #[must_use]
    pub fn match_device(&self, info: &DeviceInfo) -> Option<&str> {
        self.manifest
            .devices
            .iter()
            .find(|d| d.enabled && kind_fits(d.kind, info.kind) && matches(&d.matches, info))
            .map(|d| d.name.as_str())
    }

    /// Reloads one explicitly named camera and schedules only its worker through the normal stop/start lifecycle.
    /// Returns its actions and whether an existing stop already covered this request.
    /// # Errors
    /// Unknown, non-camera, absent, or stale physical identities are rejected without changing any worker.
    pub fn restart_camera(&mut self, name: &str, fresh: Manifest, now_ms: u64) -> Result<(Vec<Action>, bool), String> {
        let old = self
            .manifest
            .devices
            .iter()
            .position(|device| device.name == name)
            .ok_or_else(|| format!("unknown camera {name}"))?;
        let replacement = fresh
            .devices
            .into_iter()
            .find(|device| device.name == name)
            .ok_or_else(|| format!("camera {name} is absent from the current manifest"))?;
        if !replacement.enabled
            || !matches!(self.manifest.devices[old].kind, DeviceKind::Camera | DeviceKind::Rtsp)
            || replacement.kind != self.manifest.devices[old].kind
        {
            return Err(format!("device {name} is not a camera"));
        }
        let slot = self
            .slots
            .values_mut()
            .find(|slot| slot.device == name)
            .ok_or_else(|| format!("camera {name} is not currently attached"))?;
        let attached = match &slot.state {
            State::Stopping { then: Then::Readd(info), .. } => info.clone(),
            _ => slot.info.clone(),
        };
        if !matches(&replacement.matches, &attached) || !matches!(attached.kind, DeviceKind::Camera | DeviceKind::Rtsp)
        {
            return Err(format!("camera {name} no longer matches its attached physical device"));
        }
        if matches!(slot.state, State::Stopping { then: Then::Forget, .. }) {
            return Err(format!("camera {name} was detached"));
        }
        let coalesced = matches!(slot.state, State::Stopping { .. });
        self.manifest.devices[old] = replacement;
        let mut actions = Vec::new();
        match &mut slot.state {
            State::Opening { worker, .. } | State::Running { worker, .. } => {
                let worker = *worker;
                let holds_token = matches!(slot.state, State::Opening { .. });
                slot.state = State::Stopping {
                    worker,
                    then: Then::Readd(attached.clone()),
                    holds_token,
                    deadline_ms: now_ms + self.settings.stop_grace_ms,
                    killed: false,
                };
                actions.push(Action::Stop(worker));
            }
            State::Stopping { then, .. } => *then = Then::Readd(attached),
            State::Waiting { not_before_ms } => *not_before_ms = now_ms,
        }
        self.tick(now_ms, &mut actions);
        Ok((actions, coalesced))
    }

    /// Handles an event at time `now_ms`, returning what to do.
    pub fn handle(&mut self, event: Event, now_ms: u64) -> Vec<Action> {
        let mut out = Vec::new();
        match event {
            Event::Added(info) => self.added(info),
            Event::Removed(key) => {
                self.queue.retain(|k| *k != key);
                if let Some(slot) = self.slots.get_mut(&key) {
                    match slot.state {
                        State::Opening { worker, .. } | State::Running { worker, .. } => {
                            if matches!(slot.state, State::Opening { .. }) {
                                self.token_free_ms = now_ms + self.settings.settle_ms;
                            }
                            slot.state = State::Stopping {
                                worker,
                                then: Then::Forget,
                                holds_token: false,
                                deadline_ms: now_ms + self.settings.stop_grace_ms,
                                killed: false,
                            };
                            out.push(Action::Stop(worker));
                        }
                        State::Waiting { .. } => {
                            self.slots.remove(&key);
                        }
                        State::Stopping { ref mut then, .. } => {
                            // Gone again (or gone after a failed open): don't restart it.
                            *then = Then::Forget;
                        }
                    }
                }
            }
            Event::Opened(worker) => {
                if let Some(slot) = self.slot_of(worker)
                    && let State::Opening { .. } = slot.state
                {
                    slot.state = State::Running { worker, opened_ms: now_ms };
                    self.token_free_ms = now_ms + self.settings.settle_ms;
                }
            }
            Event::Exited(worker) => self.exited(worker, now_ms),
        }
        self.tick(now_ms, &mut out);
        out
    }

    /// Lets time pass: restarts that are due, a worker that took too long to open, the next start.
    pub fn tick_at(&mut self, now_ms: u64) -> Vec<Action> {
        let mut out = Vec::new();
        self.tick(now_ms, &mut out);
        out
    }

    /// The earliest time something may happen without a new event, so the driver can sleep until then. Always a
    /// time that, once reached, changes something: the driver never spins on a deadline already past.
    #[must_use]
    pub fn next_deadline_ms(&self) -> Option<u64> {
        let mut soonest: Option<u64> = None;
        let mut take = |t: u64| soonest = Some(soonest.map_or(t, |s| s.min(t)));
        for s in self.slots.values() {
            match s.state {
                State::Opening { deadline_ms, .. } => take(deadline_ms),
                State::Stopping { deadline_ms, .. } => take(deadline_ms),
                _ => {}
            }
        }
        if !self.token_held() {
            let waiting = self.queue.iter().filter_map(|k| {
                let slot = self.slots.get(k)?;
                match slot.state {
                    State::Waiting { not_before_ms } if !self.paused.contains(&slot.device) => {
                        Some(not_before_ms.max(self.token_free_ms))
                    }
                    _ => None,
                }
            });
            if let Some(t) = waiting.min() {
                take(t);
            }
        }
        soonest
    }

    /// The workers there are now, by manifest entry, for a shutdown or a health line.
    #[must_use]
    pub fn workers(&self) -> Vec<(WorkerId, &str)> {
        self.slots
            .values()
            .filter_map(|s| match s.state {
                State::Opening { worker, .. } | State::Running { worker, .. } | State::Stopping { worker, .. } => {
                    Some((worker, s.device.as_str()))
                }
                State::Waiting { .. } => None,
            })
            .collect()
    }

    fn token_held(&self) -> bool {
        self.slots
            .values()
            .any(|s| matches!(s.state, State::Opening { .. } | State::Stopping { holds_token: true, .. }))
    }

    fn added(&mut self, info: DeviceInfo) {
        let excluded = info.kind == DeviceKind::Microphone && self.manifest.audio.excludes(&info.name);
        if excluded {
            return;
        }
        let Some(device) = self.match_device(&info).map(str::to_owned) else { return };
        // One worker per manifest entry: a second node for it waits, unless the first is on its way out, in which
        // case this is the device coming back.
        if let Some(slot) = self.slots.values_mut().find(|s| s.device == device) {
            if let State::Stopping { then, .. } = &mut slot.state
                && *then != Then::Backoff
            {
                *then = Then::Readd(info);
            }
            return;
        }
        let key = info.key.clone();
        let backoff_ms = self.settings.backoff_initial_ms;
        self.slots.insert(key.clone(), Slot { device, info, state: State::Waiting { not_before_ms: 0 }, backoff_ms });
        self.queue.push_back(key);
    }

    fn slot_of(&mut self, worker: WorkerId) -> Option<&mut Slot> {
        self.slots.values_mut().find(|s| s.worker() == Some(worker))
    }

    fn exited(&mut self, worker: WorkerId, now_ms: u64) {
        let settings = self.settings;
        let Some(key) = self.slots.iter().find(|(_, s)| s.worker() == Some(worker)).map(|(k, _)| k.clone()) else {
            return;
        };
        if matches!(self.slots.get(&key).map(|s| &s.state), Some(State::Stopping { .. })) {
            self.stopped(&key, now_ms);
            return;
        }
        let Some(slot) = self.slots.get_mut(&key) else { return };
        match slot.state {
            State::Stopping { .. } => {}
            State::Opening { .. } => {
                // It held the token: free it after the settle delay, and back off.
                self.token_free_ms = now_ms + settings.settle_ms;
                slot.state = State::Waiting { not_before_ms: now_ms + slot.backoff_ms };
                slot.backoff_ms = (slot.backoff_ms * 2).min(settings.backoff_max_ms);
                self.queue.push_back(key);
            }
            State::Running { opened_ms, .. } => {
                if now_ms.saturating_sub(opened_ms) >= settings.healthy_after_ms {
                    slot.backoff_ms = settings.backoff_initial_ms;
                }
                slot.state = State::Waiting { not_before_ms: now_ms + slot.backoff_ms };
                slot.backoff_ms = (slot.backoff_ms * 2).min(settings.backoff_max_ms);
                self.queue.push_back(key);
            }
            State::Waiting { .. } => {}
        }
    }

    /// A stopping worker is gone (it exited, or it was given up on): free what it held, then forget the slot,
    /// restart it after its backoff, or start it again on the device that came back.
    fn stopped(&mut self, key: &str, now_ms: u64) {
        let settings = self.settings;
        let Some(mut slot) = self.slots.remove(key) else { return };
        let State::Stopping { then, holds_token, .. } = slot.state else {
            self.slots.insert(key.to_owned(), slot);
            return;
        };
        if holds_token {
            self.token_free_ms = now_ms + settings.settle_ms;
        }
        match then {
            Then::Forget => {}
            Then::Backoff => {
                slot.state = State::Waiting { not_before_ms: now_ms + slot.backoff_ms };
                slot.backoff_ms = (slot.backoff_ms * 2).min(settings.backoff_max_ms);
                self.slots.insert(key.to_owned(), slot);
                self.queue.push_back(key.to_owned());
            }
            Then::Readd(info) => {
                let new_key = info.key.clone();
                slot.info = info;
                slot.state = State::Waiting { not_before_ms: now_ms };
                slot.backoff_ms = settings.backoff_initial_ms;
                self.slots.insert(new_key.clone(), slot);
                self.queue.push_back(new_key);
            }
        }
    }

    fn tick(&mut self, now_ms: u64, out: &mut Vec<Action>) {
        let settings = self.settings;
        // A worker that held the token too long is stopped; it keeps the token until its exit comes back.
        for slot in self.slots.values_mut() {
            if let State::Opening { worker, deadline_ms, .. } = slot.state
                && now_ms >= deadline_ms
            {
                slot.state = State::Stopping {
                    worker,
                    then: Then::Backoff,
                    holds_token: true,
                    deadline_ms: now_ms + settings.stop_grace_ms,
                    killed: false,
                };
                out.push(Action::Stop(worker));
            }
        }
        // One that ignores the stop is killed; one that outlives the kill is given up on.
        let mut abandoned = Vec::new();
        for (key, slot) in &mut self.slots {
            if let State::Stopping { worker, deadline_ms, killed, .. } = &mut slot.state
                && now_ms >= *deadline_ms
            {
                if *killed {
                    abandoned.push(key.clone());
                } else {
                    *killed = true;
                    *deadline_ms = now_ms + settings.stop_grace_ms;
                    out.push(Action::Kill(*worker));
                }
            }
        }
        for key in abandoned {
            self.stopped(&key, now_ms);
        }
        if self.token_held() || now_ms < self.token_free_ms {
            return;
        }
        // The first in the queue whose restart is due takes the token.
        let due = self.queue.iter().position(|k| {
            self.slots.get(k).is_some_and(|s| {
                !self.paused.contains(&s.device)
                    && matches!(s.state, State::Waiting { not_before_ms } if now_ms >= not_before_ms)
            })
        });
        if let Some(i) = due
            && let Some(key) = self.queue.remove(i)
            && let Some(slot) = self.slots.get_mut(&key)
        {
            let worker = WorkerId(self.next_worker);
            self.next_worker += 1;
            // RTSP opens the transport/probes first, then waits for a complete IDR. Both phases have
            // independent bounded deadlines; permit their sum plus scheduling grace before stopping it.
            let timeout = self
                .manifest
                .devices
                .iter()
                .find(|device| device.name == slot.device)
                .and_then(|device| device.rtsp.as_ref())
                .map_or(settings.open_timeout_ms, |rtsp| {
                    settings
                        .open_timeout_ms
                        .max(rtsp.connect_timeout_ms.saturating_add(rtsp.read_timeout_ms).saturating_add(1_000))
                });
            slot.state = State::Opening { worker, deadline_ms: now_ms.saturating_add(timeout) };
            out.push(Action::Start { worker, device: slot.device.clone(), info: slot.info.clone() });
        }
    }
}

impl Slot {
    fn worker(&self) -> Option<WorkerId> {
        match self.state {
            State::Opening { worker, .. } | State::Running { worker, .. } | State::Stopping { worker, .. } => {
                Some(worker)
            }
            State::Waiting { .. } => None,
        }
    }
}

/// The enabled manifest entries marked `paused`.
#[must_use]
pub fn paused_in(manifest: &Manifest) -> BTreeSet<String> {
    manifest.devices.iter().filter(|d| d.enabled && d.paused).map(|d| d.name.clone()).collect()
}

/// Whether a hotplug device of kind `seen` can be the manifest's kind `want`: an XVF3800 array shows up as a
/// microphone, and its manifest entry says what more it is.
fn kind_fits(want: DeviceKind, seen: DeviceKind) -> bool {
    want == seen || (want == DeviceKind::Xvf3800 && seen == DeviceKind::Microphone)
}

fn matches(m: &DeviceMatch, info: &DeviceInfo) -> bool {
    let name_ok = m.name.as_ref().is_none_or(|n| info.name.to_lowercase().contains(&n.to_lowercase()));
    let usb_ok = m.usb.as_ref().is_none_or(|u| parse_usb(u).is_some() && parse_usb(u) == info.usb);
    let path_ok = m.path.as_ref().is_none_or(|p| info.node == *p || info.paths.iter().any(|q| q == p));
    name_ok && usb_ok && path_ok
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST: &str = r#"
        [supervisor]
        settle_ms = 1000
        backoff_initial_ms = 2000
        backoff_max_ms = 16000
        healthy_after_ms = 60000
        open_timeout_ms = 15000

        [[device]]
        name = "facecam"
        kind = "camera"
        match = { name = "Elgato Facecam" }

        [[device]]
        name = "cam2"
        kind = "camera"
        match = { usb = "0c45:6366" }

        [[device]]
        name = "wave3"
        kind = "microphone"
        match = { name = "Wave 3" }
    "#;

    fn sup() -> Supervisor {
        Supervisor::new(Manifest::parse(MANIFEST).unwrap())
    }

    fn dev(key: &str, kind: DeviceKind, name: &str, usb: Option<(u16, u16)>) -> DeviceInfo {
        DeviceInfo { key: key.into(), kind, name: name.into(), usb, node: format!("/dev/{key}"), paths: Vec::new() }
    }

    fn facecam() -> DeviceInfo {
        dev("video0", DeviceKind::Camera, "Elgato Facecam: Elgato Facecam", Some((0x0fd9, 0x0078)))
    }

    fn started(actions: &[Action]) -> Vec<(WorkerId, String)> {
        actions
            .iter()
            .filter_map(|a| match a {
                Action::Start { worker, device, .. } => Some((*worker, device.clone())),
                Action::Stop(_) | Action::Kill(_) => None,
            })
            .collect()
    }

    #[test]
    fn network_sources_match_independently_and_reconnect_without_restarting_the_other_camera() {
        let manifest = Manifest::parse(
            r#"
            [supervisor]
            settle_ms = 0
            backoff_initial_ms = 2000
            [[device]]
            name = "entryway"
            kind = "rtsp"
            rtsp = { url = "rtsp://fixture.invalid/one" }
            [[device]]
            name = "yard"
            kind = "rtsp"
            rtsp = { url = "rtsp://fixture.invalid/two" }
        "#,
        )
        .unwrap();
        let sources = crate::rtsp::configured_devices(&manifest);
        let mut supervisor = Supervisor::new(manifest);
        assert_eq!(
            started(&supervisor.handle(Event::Added(sources[0].clone()), 0)),
            vec![(WorkerId(1), "entryway".into())]
        );
        supervisor.handle(Event::Opened(WorkerId(1)), 1);
        assert_eq!(
            started(&supervisor.handle(Event::Added(sources[1].clone()), 2)),
            vec![(WorkerId(2), "yard".into())]
        );
        supervisor.handle(Event::Opened(WorkerId(2)), 3);
        assert!(supervisor.handle(Event::Exited(WorkerId(1)), 4).is_empty());
        assert_eq!(started(&supervisor.tick_at(2004)), vec![(WorkerId(3), "entryway".into())]);
    }

    #[test]
    fn disabled_network_camera_retains_configuration_but_never_matches_or_starts() {
        let text = r#"
            [[device]]
            name = "entryway"
            enabled = false
            kind = "rtsp"
            rtsp = { url = "rtsp://fixture.invalid/private" }
            [[device]]
            name = "yard"
            kind = "rtsp"
            rtsp = { url = "rtsp://fixture.invalid/two" }
        "#;
        let manifest = Manifest::parse(text).unwrap();
        assert_eq!(manifest.devices.len(), 2);
        assert!(!manifest.devices[0].enabled);
        assert!(manifest.devices[1].enabled);
        assert_eq!(
            crate::rtsp::configured_devices(&manifest).iter().map(|d| d.name.as_str()).collect::<Vec<_>>(),
            vec!["yard"]
        );
        let enabled = Manifest::parse(&text.replace("enabled = false", "enabled = true")).unwrap();
        let source = crate::rtsp::configured_devices(&enabled).remove(0);
        let mut supervisor = Supervisor::new(manifest);
        assert_eq!(supervisor.match_device(&source), None);
        assert!(supervisor.handle(Event::Added(source.clone()), 0).is_empty());
        let mut supervisor = Supervisor::new(enabled);
        assert_eq!(supervisor.match_device(&source), Some("entryway"));
        assert_eq!(started(&supervisor.handle(Event::Added(source), 0)), vec![(WorkerId(1), "entryway".into())]);
    }

    #[test]
    fn a_matching_device_gets_a_worker_and_others_are_ignored() {
        let mut s = sup();
        assert_eq!(started(&s.handle(Event::Added(facecam()), 0)), vec![(WorkerId(1), "facecam".into())]);
        let stranger = dev("video9", DeviceKind::Camera, "Some Webcam", Some((1, 2)));
        assert!(s.handle(Event::Added(stranger), 10).is_empty());
        let phone = dev("mic9", DeviceKind::Microphone, "iPhone Wave 3", None);
        assert_eq!(s.match_device(&phone), Some("wave3"));
        assert!(s.handle(Event::Added(phone), 20).is_empty(), "an excluded name isn't captured");
    }

    #[test]
    fn devices_open_one_at_a_time_with_a_settle_delay() {
        let mut s = sup();
        s.handle(Event::Added(facecam()), 0);
        let cam2 = dev("video2", DeviceKind::Camera, "USB Camera", Some((0x0c45, 0x6366)));
        assert!(started(&s.handle(Event::Added(cam2), 5)).is_empty(), "the facecam holds the token");
        assert!(started(&s.handle(Event::Opened(WorkerId(1)), 300)).is_empty(), "then the settle delay");
        assert_eq!(s.next_deadline_ms(), Some(1_300));
        assert!(s.tick_at(1_299).is_empty());
        assert_eq!(started(&s.tick_at(1_300)), vec![(WorkerId(2), "cam2".into())]);
    }

    #[test]
    fn quick_failures_back_off_doubling_up_to_the_cap() {
        let mut s = sup();
        s.handle(Event::Added(facecam()), 0);
        s.handle(Event::Opened(WorkerId(1)), 100);
        let mut now = 200;
        let mut waits = Vec::new();
        let mut worker = WorkerId(1);
        for _ in 0..6 {
            s.handle(Event::Exited(worker), now);
            let next = s.next_deadline_ms().unwrap();
            waits.push(next - now);
            let a = s.tick_at(next);
            worker = started(&a)[0].0;
            s.handle(Event::Opened(worker), next + 10);
            now = next + 100;
        }
        assert_eq!(waits, vec![2_000, 4_000, 8_000, 16_000, 16_000, 16_000]);
    }

    #[test]
    fn a_healthy_run_resets_the_backoff() {
        let mut s = sup();
        s.handle(Event::Added(facecam()), 0);
        s.handle(Event::Opened(WorkerId(1)), 0);
        s.handle(Event::Exited(WorkerId(1)), 1_000);
        let w = started(&s.tick_at(3_000))[0].0; // after 2 s
        s.handle(Event::Opened(w), 3_100);
        s.handle(Event::Exited(w), 3_200); // quick: the next wait would be 4 s
        let w = started(&s.tick_at(7_200))[0].0;
        s.handle(Event::Opened(w), 7_300);
        s.handle(Event::Exited(w), 7_300 + 60_000); // a healthy minute
        assert_eq!(s.next_deadline_ms(), Some(7_300 + 60_000 + 2_000), "back to the first step");
    }

    #[test]
    fn a_network_camera_that_records_then_drops_restarts_at_the_first_step_each_time() {
        // Driveway and patio reconnected 45 times an hour, each run 30 to 60 s, and every restart waited 60 s.
        let manifest = Manifest::parse(
            r#"
            [[device]]
            name = "driveway"
            kind = "rtsp"
            rtsp = { url = "rtsp://fixture.invalid/live" }
        "#,
        )
        .unwrap();
        let source = crate::rtsp::configured_devices(&manifest).remove(0);
        let mut s = Supervisor::new(manifest);
        s.handle(Event::Added(source), 0);
        let mut worker = WorkerId(1);
        let mut now = 0;
        let mut waits = Vec::new();
        for run_ms in [30_000, 40_000, 15_000, 3_000, 3_000, 20_000] {
            s.handle(Event::Opened(worker), now + 1_000); // a second to connect: not counted as recording
            now += 1_000 + run_ms;
            s.handle(Event::Exited(worker), now);
            let next = s.next_deadline_ms().unwrap();
            waits.push(next - now);
            worker = started(&s.tick_at(next))[0].0;
            now = next;
        }
        // Runs of 15 s or more restart after 2 s; quick deaths still double the wait; a recording run resets it.
        assert_eq!(waits, vec![2_000, 2_000, 2_000, 4_000, 8_000, 2_000]);
    }

    #[test]
    fn rtsp_opening_budget_covers_connection_and_first_idr_deadlines() {
        let manifest = Manifest::parse(
            r#"
            [supervisor]
            open_timeout_ms = 15000
            [[device]]
            name = "entryway"
            kind = "rtsp"
            rtsp = { url = "rtsp://fixture.invalid/live", connect_timeout_ms = 20000, read_timeout_ms = 10000 }
        "#,
        )
        .unwrap();
        let source = crate::rtsp::configured_devices(&manifest).remove(0);
        let mut supervisor = Supervisor::new(manifest);
        supervisor.handle(Event::Added(source), 0);
        assert_eq!(supervisor.next_deadline_ms(), Some(31000));
        assert!(
            supervisor.tick_at(15000).is_empty(),
            "a valid connection plus first-IDR wait outlives the USB deadline"
        );
        assert!(supervisor.tick_at(30999).is_empty());
        assert_eq!(supervisor.tick_at(31000), vec![Action::Stop(WorkerId(1))]);
    }

    #[test]
    fn a_worker_that_never_opens_is_stopped_and_frees_the_token() {
        let mut s = sup();
        s.handle(Event::Added(facecam()), 0);
        let cam2 = dev("video2", DeviceKind::Camera, "USB Camera", Some((0x0c45, 0x6366)));
        s.handle(Event::Added(cam2), 1);
        assert_eq!(s.tick_at(15_000), vec![Action::Stop(WorkerId(1))]);
        let a = s.handle(Event::Exited(WorkerId(1)), 15_050);
        assert!(started(&a).is_empty(), "the settle delay first");
        assert_eq!(
            started(&s.tick_at(16_050)),
            vec![(WorkerId(2), "cam2".into())],
            "the next device, not the stuck one"
        );
    }

    #[test]
    fn unplugging_one_leaves_the_other_and_replugging_restarts_it() {
        let mut s = sup();
        s.handle(Event::Added(facecam()), 0);
        s.handle(Event::Opened(WorkerId(1)), 100);
        let wave = dev("mic0", DeviceKind::Microphone, "Elgato Wave 3 Mono", None);
        assert_eq!(started(&s.handle(Event::Added(wave), 1_200)), vec![(WorkerId(2), "wave3".into())]);
        s.handle(Event::Opened(WorkerId(2)), 1_300);
        assert_eq!(s.handle(Event::Removed("video0".into()), 5_000), vec![Action::Stop(WorkerId(1))]);
        assert!(s.handle(Event::Exited(WorkerId(1)), 5_010).is_empty(), "a device that went isn't restarted");
        assert!(s.tick_at(100_000).is_empty(), "the microphone's worker is untouched");
        assert_eq!(started(&s.handle(Event::Added(facecam()), 200_000)), vec![(WorkerId(3), "facecam".into())]);
    }

    fn paused(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|n| (*n).to_owned()).collect()
    }

    #[test]
    fn pausing_stops_only_that_worker_and_it_stays_down_until_resumed() {
        let mut s = sup();
        s.handle(Event::Added(facecam()), 0);
        s.handle(Event::Opened(WorkerId(1)), 100);
        let wave = dev("mic0", DeviceKind::Microphone, "Elgato Wave 3 Mono", None);
        assert_eq!(started(&s.handle(Event::Added(wave), 1_200)), vec![(WorkerId(2), "wave3".into())]);
        s.handle(Event::Opened(WorkerId(2)), 1_300);
        assert_eq!(s.set_paused(paused(&["wave3"]), 5_000), vec![Action::Stop(WorkerId(2))]);
        assert!(s.handle(Event::Exited(WorkerId(2)), 5_010).is_empty(), "a paused device isn't restarted");
        assert_eq!(s.next_deadline_ms(), None, "nothing is due while it's paused: the driver never spins");
        assert!(s.tick_at(500_000).is_empty());
        assert_eq!(s.workers(), vec![(WorkerId(1), "facecam")], "the camera's worker is untouched");
        assert_eq!(started(&s.set_paused(paused(&[]), 600_000)), vec![(WorkerId(3), "wave3".into())]);
    }

    #[test]
    fn a_device_paused_in_the_manifest_never_opens_after_a_restart() {
        let text = MANIFEST.replace("kind = \"microphone\"", "kind = \"microphone\"\n        paused = true");
        let mut s = Supervisor::new(Manifest::parse(&text).unwrap());
        assert_eq!(s.paused(), &paused(&["wave3"]));
        let wave = dev("mic0", DeviceKind::Microphone, "Elgato Wave 3 Mono", None);
        assert!(s.handle(Event::Added(wave), 0).is_empty());
        assert_eq!(s.next_deadline_ms(), None);
        assert_eq!(started(&s.handle(Event::Added(facecam()), 10)), vec![(WorkerId(1), "facecam".into())]);
    }

    #[test]
    fn pausing_an_opening_device_keeps_the_token_until_it_exits() {
        let mut s = sup();
        s.handle(Event::Added(facecam()), 0);
        let cam2 = dev("video2", DeviceKind::Camera, "USB Camera", Some((0x0c45, 0x6366)));
        s.handle(Event::Added(cam2), 5);
        assert_eq!(s.set_paused(paused(&["facecam"]), 100), vec![Action::Stop(WorkerId(1))]);
        assert!(s.tick_at(5_000).is_empty(), "cam2 waits while the paused facecam may still be opening");
        assert!(s.handle(Event::Exited(WorkerId(1)), 5_100).is_empty(), "then the settle delay");
        assert_eq!(started(&s.tick_at(6_100)), vec![(WorkerId(2), "cam2".into())]);
    }

    #[test]
    fn the_same_pauses_again_change_nothing() {
        let mut s = sup();
        s.handle(Event::Added(facecam()), 0);
        s.handle(Event::Opened(WorkerId(1)), 100);
        assert!(s.set_paused(paused(&[]), 1_000).is_empty());
        assert_eq!(s.set_paused(paused(&["facecam"]), 2_000), vec![Action::Stop(WorkerId(1))]);
        assert!(s.set_paused(paused(&["facecam"]), 3_000).is_empty(), "no second stop");
    }

    #[test]
    fn only_enabled_devices_count_as_paused() {
        let text = MANIFEST
            .replace("name = \"cam2\"", "name = \"cam2\"\n        enabled = false\n        paused = true")
            .replace("name = \"facecam\"", "name = \"facecam\"\n        paused = true");
        let manifest = Manifest::parse(&text).unwrap();
        assert_eq!(paused_in(&manifest), paused(&["facecam"]));
        assert!(!Manifest::parse(MANIFEST).unwrap().devices[2].paused, "paused is off unless written");
    }

    #[test]
    fn a_device_listed_twice_by_hotplug_gets_one_worker() {
        let mut s = sup();
        s.handle(Event::Added(facecam()), 0);
        s.handle(Event::Opened(WorkerId(1)), 10);
        assert!(s.handle(Event::Added(facecam()), 2_000).is_empty());
    }

    #[test]
    fn a_device_replugged_while_its_worker_stops_is_started_again() {
        // Same key (a USB glitch) and a new key (re-enumerated on another port).
        for again in [facecam(), dev("video4", DeviceKind::Camera, "Elgato Facecam: Elgato Facecam", None)] {
            let mut s = sup();
            s.handle(Event::Added(facecam()), 0);
            s.handle(Event::Opened(WorkerId(1)), 10);
            assert_eq!(s.handle(Event::Removed("video0".into()), 5_000), vec![Action::Stop(WorkerId(1))]);
            assert!(s.handle(Event::Added(again.clone()), 5_100).is_empty(), "the old worker is still stopping");
            let a = s.handle(Event::Exited(WorkerId(1)), 5_200);
            assert_eq!(started(&a), vec![(WorkerId(2), "facecam".into())], "{again:?}");
            let Action::Start { info, .. } = &a[0] else { unreachable!() };
            assert_eq!(info.key, again.key, "on the device that came back");
        }
    }

    #[test]
    fn a_replug_that_goes_again_before_the_exit_is_forgotten() {
        let mut s = sup();
        s.handle(Event::Added(facecam()), 0);
        s.handle(Event::Opened(WorkerId(1)), 10);
        s.handle(Event::Removed("video0".into()), 5_000);
        s.handle(Event::Added(facecam()), 5_100);
        s.handle(Event::Removed("video0".into()), 5_150);
        assert!(s.handle(Event::Exited(WorkerId(1)), 5_200).is_empty());
        assert!(s.tick_at(1_000_000).is_empty());
        assert_eq!(s.next_deadline_ms(), None);
    }

    #[test]
    fn a_worker_that_ignores_the_stop_is_killed_then_given_up_on() {
        let mut s = sup();
        s.handle(Event::Added(facecam()), 0);
        s.handle(Event::Added(dev("video2", DeviceKind::Camera, "Cam", Some((0x0c45, 0x6366)))), 0);
        // Facecam never opens: stopped at the timeout, and it keeps the token (cam2 doesn't start).
        assert_eq!(s.tick_at(15_000), vec![Action::Stop(WorkerId(1))]);
        // The kill comes stop_grace_ms (10 s) later.
        assert_eq!(s.next_deadline_ms(), Some(25_000), "the kill, not a deadline in the past");
        assert!(s.tick_at(24_999).is_empty());
        assert_eq!(s.tick_at(25_000), vec![Action::Kill(WorkerId(1))]);
        assert_eq!(s.next_deadline_ms(), Some(35_000));
        // Even the kill doesn't end it (stuck in the kernel): given up on, the token frees after the settle, cam2
        // starts, and the facecam retries after its backoff.
        assert!(s.tick_at(35_000).is_empty());
        assert_eq!(s.next_deadline_ms(), Some(36_000));
        assert_eq!(started(&s.tick_at(36_000)), vec![(WorkerId(2), "cam2".into())]);
        s.handle(Event::Opened(WorkerId(2)), 36_100);
        assert_eq!(started(&s.tick_at(37_100)), vec![(WorkerId(3), "facecam".into())]);
        // The stuck worker's late exit changes nothing.
        assert!(s.handle(Event::Exited(WorkerId(1)), 40_000).is_empty());
    }

    #[test]
    fn the_next_deadline_is_never_in_the_past() {
        let mut s = sup();
        s.handle(Event::Added(facecam()), 0);
        let mut now = 0;
        for _ in 0..20 {
            let Some(next) = s.next_deadline_ms() else { break };
            assert!(next >= now, "{next} before {now}");
            now = next;
            s.tick_at(now);
        }
    }

    #[test]
    fn a_second_node_for_the_same_manifest_entry_gets_no_worker() {
        let mut s = sup();
        s.handle(Event::Added(facecam()), 0);
        let meta = dev("video1", DeviceKind::Camera, "Elgato Facecam: Elgato Facecam", Some((0x0fd9, 0x0078)));
        assert!(s.handle(Event::Added(meta), 10).is_empty());
        s.handle(Event::Opened(WorkerId(1)), 20);
        assert!(s.tick_at(100_000).is_empty());
        assert_eq!(s.workers(), vec![(WorkerId(1), "facecam")]);
    }

    #[test]
    fn unplugging_a_device_while_it_opens_frees_the_token_after_the_settle() {
        let mut s = sup();
        s.handle(Event::Added(facecam()), 0);
        s.handle(Event::Added(dev("video2", DeviceKind::Camera, "Cam", Some((0x0c45, 0x6366)))), 0);
        assert_eq!(s.handle(Event::Removed("video0".into()), 500), vec![Action::Stop(WorkerId(1))]);
        assert_eq!(started(&s.tick_at(1_500)), vec![(WorkerId(2), "cam2".into())]);
        assert!(s.handle(Event::Exited(WorkerId(1)), 1_600).is_empty(), "no restart for a device that went");
    }
    #[test]
    fn an_explicit_camera_restart_leaves_other_cameras_and_microphones_alone_and_coalesces() {
        let text = "[supervisor]\nsettle_ms = 0\n[[device]]\nname = 'cam0'\nkind = 'camera'\nmatch = {name = 'First'}\n[[device]]\nname = 'cam1'\nkind = 'camera'\nmatch = {name = 'Second'}\n[[device]]\nname = 'mic0'\nkind = 'microphone'\nmatch = {name = 'Microphone'}\n";
        let manifest = Manifest::parse(text).unwrap();
        let mut supervisor = Supervisor::new(manifest.clone());
        for (index, (name, kind)) in
            [("First", DeviceKind::Camera), ("Second", DeviceKind::Camera), ("Microphone", DeviceKind::Microphone)]
                .into_iter()
                .enumerate()
        {
            let info = DeviceInfo {
                key: name.into(),
                name: name.into(),
                node: format!("fake-{index}"),
                kind,
                usb: None,
                paths: vec![],
            };
            assert!(matches!(&supervisor.handle(Event::Added(info), 0)[..], [Action::Start { .. }]));
            supervisor.handle(Event::Opened(WorkerId(index as u64 + 1)), 0);
        }
        let fresh =
            Manifest::parse(&text.replace("name = 'cam0'", "name = 'cam0'\nvideo = {width = 640, height = 480}"))
                .unwrap();
        let (actions, coalesced) = supervisor.restart_camera("cam0", fresh.clone(), 1).unwrap();
        assert_eq!(actions, vec![Action::Stop(WorkerId(1))]);
        assert!(!coalesced);
        let (actions, coalesced) = supervisor.restart_camera("cam0", fresh, 2).unwrap();
        assert!(actions.is_empty());
        assert!(coalesced);
        let restarted = supervisor.handle(Event::Exited(WorkerId(1)), 3);
        assert!(matches!(&restarted[..], [Action::Start { worker: WorkerId(4), device, .. }] if device == "cam0"));
        assert!(supervisor.workers().contains(&(WorkerId(2), "cam1")));
        assert!(supervisor.workers().contains(&(WorkerId(3), "mic0")));
        assert!(supervisor.restart_camera("mic0", manifest.clone(), 4).is_err());
        assert!(supervisor.restart_camera("missing", manifest.clone(), 4).is_err());
        let stale = Manifest::parse(&text.replace("name = 'First'", "name = 'Other physical camera'")).unwrap();
        assert!(supervisor.restart_camera("cam0", stale, 4).is_err());
        assert_eq!(supervisor.workers().len(), 3);
    }
    #[test]
    fn fable_regression_a_restart_during_replug_preserves_the_new_physical_node() {
        let mut supervisor = sup();
        supervisor.handle(Event::Added(facecam()), 0);
        supervisor.handle(Event::Opened(WorkerId(1)), 10);
        supervisor.handle(Event::Removed("video0".into()), 5000);
        let mut attached = facecam();
        attached.key = "video2".into();
        attached.node = "/dev/video2".into();
        supervisor.handle(Event::Added(attached.clone()), 5100);
        let (actions, coalesced) =
            supervisor.restart_camera("facecam", Manifest::parse(MANIFEST).unwrap(), 5150).unwrap();
        assert!(actions.is_empty() && coalesced);
        let actions = supervisor.handle(Event::Exited(WorkerId(1)), 5200);
        assert!(
            matches!(&actions[..], [Action::Start { info, device, .. }] if info == &attached && device == "facecam")
        );
    }
}
