//! The supervisor's driver: it feeds the [`Supervisor`] its events (hotplug, a worker's `opened`, a worker's exit,
//! time) and carries out its actions on worker processes. The processes are behind [`Launcher`], so the loop is
//! tested with a fake; [`ProcessLauncher`] runs `continuum-capture-worker`.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::supervisor::{Action, DeviceInfo, Event, Supervisor, WorkerId};

/// How often the supervisor reads the manifest's privacy pauses.
const PAUSE_POLL: Duration = Duration::from_secs(1);

/// Starting and signalling workers. A started worker's `opened` and its exit come back as events on the driver's
/// channel.
pub trait Launcher {
    /// Starts a worker for manifest entry `device` on `info`. An error means it never started: the driver reports
    /// it as exited at once.
    ///
    /// # Errors
    ///
    /// Why it couldn't start.
    fn start(&mut self, worker: WorkerId, device: &str, info: &DeviceInfo) -> std::io::Result<()>;
    /// Asks a worker to stop (`SIGTERM`), or, with `kill`, ends it (`SIGKILL`).
    fn signal(&mut self, worker: WorkerId, kill: bool);
    /// Processes still owned by the launcher, including any the supervisor stopped tracking after a timeout.
    fn owned_workers(&self) -> Vec<WorkerId> {
        Vec::new()
    }
}

/// The supervisor and its launcher, driven by events.
pub struct Driver<L: Launcher> {
    supervisor: Supervisor,
    launcher: L,
    events: Sender<Event>,
    /// The last pause poll couldn't read the manifest: said once until it reads again.
    pause_read_failed: bool,
    /// Where to say which pauses are in effect (`capture_paused.json` beside the control socket), and what it last
    /// said, so it's rewritten only on a change.
    pause_status: Option<(PathBuf, Option<String>)>,
}

impl<L: Launcher> Driver<L> {
    /// A driver whose launcher reports on `events` (the sending side of the channel [`Driver::run`] reads).
    pub fn new(supervisor: Supervisor, launcher: L, events: Sender<Event>) -> Self {
        Self { supervisor, launcher, events, pause_read_failed: false, pause_status: None }
    }

    /// Handles one event (or, with `None`, just the passing of time) at `now_ms`, carrying out what follows.
    pub fn step(&mut self, event: Option<Event>, now_ms: u64) {
        let actions = match event {
            Some(e) => self.supervisor.handle(e, now_ms),
            None => self.supervisor.tick_at(now_ms),
        };
        self.actions(actions);
    }

    fn actions(&mut self, actions: Vec<Action>) {
        for action in actions {
            match action {
                Action::Start { worker, device, info } => {
                    eprintln!("capture supervisor: starting worker {} for {device} on {}", worker.0, info.node);
                    if let Err(e) = self.launcher.start(worker, &device, &info) {
                        eprintln!("capture supervisor: worker {} for {device} didn't start: {e}", worker.0);
                        let _ = self.events.send(Event::Exited(worker));
                    }
                }
                Action::Stop(worker) => self.launcher.signal(worker, false),
                Action::Kill(worker) => {
                    eprintln!("capture supervisor: worker {} ignored the stop: killing it", worker.0);
                    self.launcher.signal(worker, true);
                }
            }
        }
    }

    fn restart_request(&mut self, request: crate::control::RestartRequest, manifest: &std::path::Path, now_ms: u64) {
        let result = std::fs::read_to_string(manifest)
            .map_err(|error| error.to_string())
            .and_then(|text| crate::manifest::Manifest::parse(&text).map_err(|error| error.to_string()))
            .and_then(|fresh| self.supervisor.restart_camera(&request.device, fresh, now_ms));
        let reply = match result {
            Ok((actions, coalesced)) => {
                self.actions(actions);
                crate::control::Reply::new(
                    "restarting",
                    &request.device,
                    if coalesced {
                        "camera restart already in progress; settings reloaded"
                    } else {
                        "camera restart scheduled"
                    },
                )
            }
            Err(error) => crate::control::Reply::new("failed", &request.device, error),
        };
        let _ = request.reply.send(reply);
    }

    /// Also writes the pauses in effect to `path`, so the dashboard can tell a saved pause from an applied one.
    #[must_use]
    pub fn with_pause_status(mut self, path: PathBuf) -> Self {
        self.pause_status = Some((path, None));
        self
    }

    /// Writes `{"paused": [...], "stopping": [...]}`: the paused manifest entries, and those of them whose worker
    /// hasn't exited yet. Only on a change, atomically.
    fn write_pause_status(&mut self) {
        let Some((path, last)) = &mut self.pause_status else { return };
        let paused = self.supervisor.paused();
        let stopping: Vec<&str> =
            self.supervisor.workers().into_iter().map(|(_, d)| d).filter(|d| paused.contains(*d)).collect();
        let text = serde_json::json!({"paused": paused, "stopping": stopping}).to_string();
        if last.as_deref() == Some(text.as_str()) {
            return;
        }
        let tmp = path.with_extension("json.tmp");
        match std::fs::write(&tmp, &text).and_then(|()| std::fs::rename(&tmp, &*path)) {
            Ok(()) => *last = Some(text),
            Err(error) => eprintln!("capture supervisor: WARN can't write {}: {error}", path.display()),
        }
    }

    /// Follows the manifest's `paused` flags: reads it again and applies a changed paused set. A manifest that
    /// can't be read or parsed keeps the current set, so a bad edit never resumes a paused device.
    fn follow_pauses(&mut self, manifest: &std::path::Path, now_ms: u64) {
        let fresh = match std::fs::read_to_string(manifest)
            .map_err(|error| error.to_string())
            .and_then(|text| crate::manifest::Manifest::parse(&text).map_err(|error| error.to_string()))
        {
            Ok(fresh) => fresh,
            Err(error) => {
                if !self.pause_read_failed {
                    eprintln!("capture supervisor: WARN can't read pauses from the manifest: {error}");
                }
                self.pause_read_failed = true;
                return;
            }
        };
        self.pause_read_failed = false;
        let paused = crate::supervisor::paused_in(&fresh);
        if &paused == self.supervisor.paused() {
            self.write_pause_status();
            return;
        }
        let before = self.supervisor.paused().clone();
        for name in paused.difference(&before) {
            eprintln!("capture supervisor: {name} paused for privacy: stopping its capture");
        }
        for name in before.difference(&paused) {
            eprintln!("capture supervisor: {name} resumed");
        }
        let actions = self.supervisor.set_paused(paused, now_ms);
        self.actions(actions);
        self.write_pause_status();
    }

    /// Runs until `stop` is set, then stops every worker and waits for them (up to `grace`, then kills the rest).
    pub fn run(&mut self, events: &Receiver<Event>, stop: &AtomicBool, grace: Duration) {
        self.run_loop(events, stop, grace, None);
    }

    /// Runs with a private control queue. Each request reloads the configured manifest on this supervisor thread;
    /// requests cannot supply a different manifest pathname or signal processes themselves.
    #[cfg(unix)]
    pub fn run_controlled(
        &mut self,
        events: &Receiver<Event>,
        requests: &Receiver<crate::control::RestartRequest>,
        manifest: &std::path::Path,
        stop: &AtomicBool,
        grace: Duration,
    ) {
        self.run_loop(events, stop, grace, Some((requests, manifest)));
    }

    fn run_loop(
        &mut self,
        events: &Receiver<Event>,
        stop: &AtomicBool,
        grace: Duration,
        control: Option<(&Receiver<crate::control::RestartRequest>, &std::path::Path)>,
    ) {
        let start = Instant::now();
        let now = || u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
        let mut pauses_checked: Option<Instant> = None;
        while !stop.load(Ordering::SeqCst) {
            if let Some((requests, manifest)) = control {
                for request in requests.try_iter().take(8) {
                    self.restart_request(request, manifest, now());
                }
                if pauses_checked.is_none_or(|t| t.elapsed() >= PAUSE_POLL) {
                    pauses_checked = Some(Instant::now());
                    self.follow_pauses(manifest, now());
                }
            }
            // Wake for the next deadline, and at least every 200 ms to notice `stop`.
            let wait = self.supervisor.next_deadline_ms().map_or(200, |d| d.saturating_sub(now()).min(200));
            match events.recv_timeout(Duration::from_millis(wait)) {
                Ok(e) => self.step(Some(e), now()),
                Err(RecvTimeoutError::Timeout) => self.step(None, now()),
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        let mut running: Vec<WorkerId> = self.supervisor.workers().into_iter().map(|(w, _)| w).collect();
        running.extend(self.launcher.owned_workers());
        running.sort_unstable();
        running.dedup();
        for &w in &running {
            self.launcher.signal(w, false);
        }
        let until = Instant::now() + grace;
        while !running.is_empty() && Instant::now() < until {
            if let Ok(Event::Exited(w)) = events.recv_timeout(Duration::from_millis(50)) {
                running.retain(|r| *r != w);
            }
        }
        for &w in &running {
            self.launcher.signal(w, true);
        }
        // SIGKILL only requests termination. Wait for the launcher's post-reap event before exiting, so its
        // reader thread can release the PID and no worker is deliberately left behind by supervisor shutdown.
        let until = Instant::now() + grace;
        while !running.is_empty() && Instant::now() < until {
            if let Ok(Event::Exited(w)) = events.recv_timeout(Duration::from_millis(50)) {
                running.retain(|r| *r != w);
            }
        }
        if !running.is_empty() {
            eprintln!(
                "capture supervisor: ERROR workers did not exit after SIGKILL: {running:?}; check OS device state before restarting capture"
            );
        }
    }
}

/// Runs each worker as a `continuum-capture-worker` process.
pub struct ProcessLauncher {
    /// The worker program.
    pub program: PathBuf,
    /// The manifest it reads.
    pub manifest: PathBuf,
    /// The broker's socket.
    pub socket: PathBuf,
    events: Sender<Event>,
    /// Live workers' pids: a worker's reader thread removes its own once the process is reaped, so a signal never
    /// reaches a reused pid.
    pids: Arc<Mutex<BTreeMap<WorkerId, u32>>>,
    mac_raw_share: Option<u64>,
}

impl ProcessLauncher {
    /// A launcher running `program`, reporting on `events`.
    #[must_use]
    pub fn new(program: PathBuf, manifest: PathBuf, socket: PathBuf, events: Sender<Event>) -> Self {
        Self { program, manifest, socket, events, pids: Arc::new(Mutex::new(BTreeMap::new())), mac_raw_share: None }
    }

    /// Fixes every camera's bandwidth share for this supervisor's lifetime, including later hotplug/restarts.
    #[must_use]
    pub fn with_mac_raw_share(mut self, bytes_per_second: u64) -> Self {
        self.mac_raw_share = Some(bytes_per_second);
        self
    }
}

impl Launcher for ProcessLauncher {
    fn start(&mut self, worker: WorkerId, device: &str, info: &DeviceInfo) -> std::io::Result<()> {
        let mut command = Command::new(&self.program);
        if let Some(share) = self.mac_raw_share {
            command.arg("--mac-raw-share").arg(share.to_string());
        }
        let mut child: Child = command
            .arg("--manifest")
            .arg(&self.manifest)
            .args(["--device", device, "--node", &info.node])
            .arg("--socket")
            .arg(&self.socket)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        self.pids.lock().unwrap_or_else(std::sync::PoisonError::into_inner).insert(worker, child.id());
        let stdout = child.stdout.take();
        // Reap independently of stdout: a descendant may inherit the pipe after the direct worker exits.
        let child = Arc::new(Mutex::new(child));
        let reaping = Arc::clone(&child);
        let events = self.events.clone();
        let pids = Arc::clone(&self.pids);
        if let Err(error) = std::thread::Builder::new().name(format!("worker {} reap", worker.0)).spawn(move || {
            let mut child = reaping.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            // Reserve the PID as a zombie until wait and map removal can happen under the same lock.
            wait_exited(child.id());
            let status = {
                let mut pids = pids.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                let status = child.wait();
                pids.remove(&worker);
                status
            };
            eprintln!("capture supervisor: worker {} ended: {status:?}", worker.0);
            let _ = events.send(Event::Exited(worker));
        }) {
            let mut child = child.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let mut pids = self.pids.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let _ = child.kill();
            let _ = child.wait();
            pids.remove(&worker);
            return Err(error);
        }
        let events = self.events.clone();
        if let Err(error) = std::thread::Builder::new().name(format!("worker {} stdout", worker.0)).spawn(move || {
            if let Some(out) = stdout {
                for line in BufReader::new(out).lines() {
                    match line {
                        Ok(l) if l.trim() == "opened" => {
                            let _ = events.send(Event::Opened(worker));
                        }
                        Ok(_) => {}
                        Err(_) => break,
                    }
                }
            }
        }) {
            // The reaper is already active, so this failure cannot strand the child or its PID registration.
            self.signal(worker, true);
            return Err(error);
        }
        Ok(())
    }

    fn signal(&mut self, worker: WorkerId, kill: bool) {
        let pids = self.pids.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(&pid) = pids.get(&worker) else { return };
        let Ok(pid) = libc::pid_t::try_from(pid) else { return };
        // SAFETY: kill(2) on a pid this launcher started and hasn't reaped (the map's lock is held, and a reaped
        // worker's pid is removed under it), so it can't have been reused.
        unsafe {
            libc::kill(pid, if kill { libc::SIGKILL } else { libc::SIGTERM });
        }
    }

    fn owned_workers(&self) -> Vec<WorkerId> {
        self.pids.lock().unwrap_or_else(std::sync::PoisonError::into_inner).keys().copied().collect()
    }
}

/// Blocks until process `pid` (a child) has exited, leaving it to be reaped.
fn wait_exited(pid: u32) {
    let pid: libc::id_t = pid;
    loop {
        // SAFETY: a zeroed siginfo_t is a valid out parameter for waitid.
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        // SAFETY: waitid on our own child with WNOWAIT leaves it unreaped; `info` is valid for writing.
        let r = unsafe { libc::waitid(libc::P_PID, pid, &raw mut info, libc::WEXITED | libc::WNOWAIT) };
        if r == 0 || std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{DeviceKind, Manifest};
    use std::sync::mpsc::channel;

    #[derive(Default)]
    struct Fake {
        log: Vec<String>,
        refuse: bool,
    }

    impl Launcher for Fake {
        fn start(&mut self, worker: WorkerId, device: &str, _: &DeviceInfo) -> std::io::Result<()> {
            self.log.push(format!("start {} {device}", worker.0));
            if self.refuse { Err(std::io::Error::other("no such program")) } else { Ok(()) }
        }
        fn signal(&mut self, worker: WorkerId, kill: bool) {
            self.log.push(format!("{} {}", if kill { "kill" } else { "term" }, worker.0));
        }
    }

    fn manifest() -> Manifest {
        Manifest::parse("[[device]]\nname = \"facecam\"\nkind = \"camera\"\nmatch = { name = \"Facecam\" }\n").unwrap()
    }

    fn facecam() -> DeviceInfo {
        DeviceInfo {
            key: "/devices/x/video0".into(),
            kind: DeviceKind::Camera,
            name: "Elgato Facecam".into(),
            usb: None,
            node: "/dev/video0".into(),
            paths: vec![],
        }
    }

    #[test]
    fn a_worker_that_never_starts_counts_as_exited_and_is_retried_after_backoff() {
        let (tx, rx) = channel();
        let mut d = Driver::new(Supervisor::new(manifest()), Fake { refuse: true, ..Fake::default() }, tx);
        d.step(Some(Event::Added(facecam())), 0);
        assert_eq!(d.launcher.log, vec!["start 1 facecam"]);
        let exited = rx.try_recv().unwrap();
        assert_eq!(exited, Event::Exited(WorkerId(1)));
        d.step(Some(exited), 10);
        d.step(None, 1_000);
        assert_eq!(d.launcher.log.len(), 1, "it waits out the backoff");
        d.step(None, 2_010);
        assert_eq!(d.launcher.log, vec!["start 1 facecam", "start 2 facecam"]);
    }

    #[test]
    fn the_manifests_pause_flag_stops_and_restarts_the_worker_and_a_bad_manifest_keeps_it_paused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("devices.toml");
        let text = "[[device]]\nname = \"facecam\"\nkind = \"camera\"\nmatch = { name = \"Facecam\" }\n";
        let (tx, _rx) = channel();
        let status = dir.path().join("capture_paused.json");
        let said = || std::fs::read_to_string(&status).unwrap();
        let mut d = Driver::new(Supervisor::new(manifest()), Fake::default(), tx).with_pause_status(status.clone());
        d.step(Some(Event::Added(facecam())), 0);
        d.step(Some(Event::Opened(WorkerId(1))), 100);
        std::fs::write(&path, text.replace("kind = \"camera\"", "kind = \"camera\"\npaused = true")).unwrap();
        d.follow_pauses(&path, 1_000);
        assert_eq!(d.launcher.log, vec!["start 1 facecam", "term 1"]);
        assert_eq!(said(), r#"{"paused":["facecam"],"stopping":["facecam"]}"#);
        d.step(Some(Event::Exited(WorkerId(1))), 1_100);
        d.follow_pauses(&path, 1_200);
        assert_eq!(said(), r#"{"paused":["facecam"],"stopping":[]}"#, "applied once its worker is gone");
        std::fs::write(&path, "not a manifest [").unwrap();
        d.follow_pauses(&path, 2_000);
        d.step(None, 100_000);
        assert_eq!(d.launcher.log, vec!["start 1 facecam", "term 1"], "an unreadable manifest never resumes it");
        std::fs::write(&path, text).unwrap();
        d.follow_pauses(&path, 101_000);
        assert_eq!(d.launcher.log, vec!["start 1 facecam", "term 1", "start 2 facecam"]);
        assert_eq!(said(), r#"{"paused":[],"stopping":[]}"#);
    }

    #[test]
    fn stopping_the_driver_stops_its_workers_and_kills_those_that_linger() {
        let (tx, rx) = channel();
        let mut d = Driver::new(Supervisor::new(manifest()), Fake::default(), tx.clone());
        d.step(Some(Event::Added(facecam())), 0);
        d.step(Some(Event::Opened(WorkerId(1))), 100);
        let stop = AtomicBool::new(true);
        d.run(&rx, &stop, Duration::from_millis(100));
        assert_eq!(d.launcher.log, vec!["start 1 facecam", "term 1", "kill 1"]);
        // And one that exits in time isn't killed.
        let (tx, rx) = channel();
        let mut d = Driver::new(Supervisor::new(manifest()), Fake::default(), tx.clone());
        d.step(Some(Event::Added(facecam())), 0);
        tx.send(Event::Exited(WorkerId(1))).unwrap();
        d.run(&rx, &stop, Duration::from_secs(5));
        assert_eq!(d.launcher.log, vec!["start 1 facecam", "term 1"]);
    }

    #[test]
    fn a_mixed_shutdown_kills_only_the_worker_that_lingers() {
        let two = Manifest::parse(
            "[[device]]\nname = \"facecam\"\nkind = \"camera\"\nmatch = { name = \"Facecam\" }\n\
             [[device]]\nname = \"wave3\"\nkind = \"microphone\"\nmatch = { name = \"Wave\" }\n",
        )
        .unwrap();
        let (tx, rx) = channel();
        let mut d = Driver::new(Supervisor::new(two), Fake::default(), tx.clone());
        d.step(Some(Event::Added(facecam())), 0);
        d.step(Some(Event::Opened(WorkerId(1))), 100);
        let wave = DeviceInfo {
            key: "pipewire:57".into(),
            kind: DeviceKind::Microphone,
            name: "Wave:3 Mono".into(),
            usb: None,
            node: "alsa_input.wave3".into(),
            paths: vec![],
        };
        d.step(Some(Event::Added(wave)), 2_000);
        d.step(Some(Event::Opened(WorkerId(2))), 2_100);
        // During the grace: the microphone exits, a stray Opened arrives, and the camera never exits.
        tx.send(Event::Opened(WorkerId(1))).unwrap();
        tx.send(Event::Exited(WorkerId(2))).unwrap();
        d.run(&rx, &AtomicBool::new(true), Duration::from_millis(200));
        assert_eq!(d.launcher.log, vec!["start 1 facecam", "start 2 wave3", "term 1", "term 2", "kill 1"]);
    }

    #[test]
    fn shutdown_waits_for_the_killed_worker_to_be_reaped() {
        struct Reaping {
            killed: Sender<()>,
        }
        impl Launcher for Reaping {
            fn start(&mut self, _: WorkerId, _: &str, _: &DeviceInfo) -> std::io::Result<()> {
                Ok(())
            }
            fn signal(&mut self, _: WorkerId, kill: bool) {
                if kill {
                    self.killed.send(()).unwrap();
                }
            }
        }
        let (tx, rx) = channel();
        let (killed, kill_seen) = channel();
        let (finished, done) = channel();
        let mut driver = Driver::new(Supervisor::new(manifest()), Reaping { killed }, tx.clone());
        driver.step(Some(Event::Added(facecam())), 0);
        let task = std::thread::spawn(move || {
            driver.run(&rx, &AtomicBool::new(true), Duration::from_millis(200));
            finished.send(()).unwrap();
        });
        kill_seen.recv_timeout(Duration::from_secs(5)).unwrap();
        let early = done.recv_timeout(Duration::from_millis(50));
        // Exited is emitted only after ProcessLauncher has reaped and removed the worker's PID.
        let _ = tx.send(Event::Exited(WorkerId(1)));
        task.join().unwrap();
        assert!(matches!(early, Err(RecvTimeoutError::Timeout)), "shutdown returned before reap confirmation");
        done.recv_timeout(Duration::from_secs(5)).unwrap();
    }

    #[test]
    fn shutdown_includes_children_the_supervisor_no_longer_tracks() {
        struct Owned(Fake);
        impl Launcher for Owned {
            fn start(&mut self, worker: WorkerId, name: &str, info: &DeviceInfo) -> std::io::Result<()> {
                self.0.start(worker, name, info)
            }
            fn signal(&mut self, worker: WorkerId, kill: bool) {
                self.0.signal(worker, kill);
            }
            fn owned_workers(&self) -> Vec<WorkerId> {
                vec![WorkerId(7)]
            }
        }
        let (tx, rx) = channel();
        let mut driver = Driver::new(Supervisor::new(manifest()), Owned(Fake::default()), tx);
        driver.run(&rx, &AtomicBool::new(true), Duration::ZERO);
        assert_eq!(driver.launcher.0.log, ["term 7", "kill 7"]);
    }
    #[test]
    fn a_control_request_rereads_the_manifest_before_touching_the_target() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("devices.toml");
        let original = "[[device]]\nname = 'facecam'\nkind = 'camera'\nmatch = {name = 'Facecam'}\n";
        std::fs::write(&path, original).unwrap();
        let (tx, _) = channel();
        let mut driver = Driver::new(Supervisor::new(manifest()), Fake::default(), tx);
        driver.step(Some(Event::Added(facecam())), 0);
        driver.step(Some(Event::Opened(WorkerId(1))), 1);
        std::fs::write(&path, "invalid manifest [").unwrap();
        let (send, receive) = tokio::sync::oneshot::channel();
        driver.restart_request(crate::control::RestartRequest { device: "facecam".into(), reply: send }, &path, 2);
        assert_eq!(receive.blocking_recv().unwrap().status, "failed");
        assert_eq!(driver.launcher.log, ["start 1 facecam"]);
        std::fs::write(
            &path,
            original.replace("kind = 'camera'", "kind = 'camera'\nvideo = {width = 640, height = 480}"),
        )
        .unwrap();
        let (send, receive) = tokio::sync::oneshot::channel();
        driver.restart_request(crate::control::RestartRequest { device: "facecam".into(), reply: send }, &path, 3);
        let reply = receive.blocking_recv().unwrap();
        assert_eq!((reply.status.as_str(), reply.device.as_str()), ("restarting", "facecam"));
        assert_eq!(driver.launcher.log, ["start 1 facecam", "term 1"]);
    }
}
