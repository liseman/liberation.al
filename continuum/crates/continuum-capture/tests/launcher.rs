//! The process launcher against a stand-in worker (a shell script), not a device: a worker's `opened` line and its
//! exit come back as events, and a stop or a kill reaches the process.

use std::os::unix::fs::PermissionsExt;
use std::sync::mpsc::channel;
use std::time::Duration;

use continuum_capture::driver::{Launcher, ProcessLauncher};
use continuum_capture::manifest::DeviceKind;
use continuum_capture::supervisor::{DeviceInfo, Event, WorkerId};

/// Held while a test writes its script and starts it. Another test's fork between a script's write and its exec
/// would inherit the open file, and the exec would fail with "Text file busy" (ETXTBSY), so the two happen together.
static SPAWN: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Writes the stand-in worker script and starts it, under [`SPAWN`].
fn start(l: &mut ProcessLauncher, worker: WorkerId, device: &str) {
    let _held = SPAWN.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    l.start(worker, device, &info()).unwrap();
}

fn script(dir: &std::path::Path, body: &str) -> std::path::PathBuf {
    let _held = SPAWN.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let path = dir.join("worker.sh");
    std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

fn info() -> DeviceInfo {
    DeviceInfo {
        key: "k".into(),
        kind: DeviceKind::Camera,
        name: "cam".into(),
        usb: None,
        node: "/dev/null".into(),
        paths: vec![],
    }
}

#[test]
fn opened_and_exit_come_back_and_a_stop_reaches_the_worker() {
    let dir = tempfile::tempdir().unwrap();
    // It says it opened, then waits; SIGTERM ends it (sh's default).
    let program = script(dir.path(), "echo opened\nexec sleep 30");
    let (tx, rx) = channel();
    let mut l = ProcessLauncher::new(program, "m.toml".into(), "b.sock".into(), tx);
    start(&mut l, WorkerId(1), "cam");
    assert_eq!(rx.recv_timeout(Duration::from_secs(10)).unwrap(), Event::Opened(WorkerId(1)));
    l.signal(WorkerId(1), false);
    assert_eq!(rx.recv_timeout(Duration::from_secs(10)).unwrap(), Event::Exited(WorkerId(1)));
}

#[test]
fn a_worker_that_ignores_the_stop_is_killed() {
    let dir = tempfile::tempdir().unwrap();
    let program = script(dir.path(), "trap '' TERM\necho opened\nwhile :; do sleep 1; done");
    let (tx, rx) = channel();
    let mut l = ProcessLauncher::new(program, "m.toml".into(), "b.sock".into(), tx);
    start(&mut l, WorkerId(7), "cam");
    assert_eq!(rx.recv_timeout(Duration::from_secs(10)).unwrap(), Event::Opened(WorkerId(7)));
    l.signal(WorkerId(7), false);
    assert!(rx.recv_timeout(Duration::from_millis(500)).is_err(), "it ignored the stop");
    l.signal(WorkerId(7), true);
    assert_eq!(rx.recv_timeout(Duration::from_secs(10)).unwrap(), Event::Exited(WorkerId(7)));
}

#[test]
fn a_worker_that_closes_its_stdout_can_still_be_stopped() {
    let dir = tempfile::tempdir().unwrap();
    let program = script(dir.path(), "echo opened\nexec >&-\nexec sleep 30");
    let (tx, rx) = channel();
    let mut l = ProcessLauncher::new(program, "m.toml".into(), "b.sock".into(), tx);
    start(&mut l, WorkerId(3), "cam");
    assert_eq!(rx.recv_timeout(Duration::from_secs(10)).unwrap(), Event::Opened(WorkerId(3)));
    std::thread::sleep(Duration::from_millis(200)); // its reader has seen the end of stdout and is waiting
    l.signal(WorkerId(3), false);
    assert_eq!(rx.recv_timeout(Duration::from_secs(10)).unwrap(), Event::Exited(WorkerId(3)));
    l.signal(WorkerId(3), true); // reaped and forgotten: nothing to signal
}

#[test]
fn child_reaping_does_not_wait_for_a_descendants_inherited_stdout() {
    let dir = tempfile::tempdir().unwrap();
    // The worker itself exits immediately. Its bounded stand-in descendant holds the stdout pipe open.
    let program = script(dir.path(), "sleep 3 &\necho opened\nexit 0");
    let (tx, rx) = channel();
    let mut launcher = ProcessLauncher::new(program, "m.toml".into(), "b.sock".into(), tx);
    start(&mut launcher, WorkerId(5), "cam");
    let first = rx.recv_timeout(Duration::from_secs(5)).unwrap();
    // Reader/reaper threads are independent, so either event can arrive first.
    let reaped = if first == Event::Exited(WorkerId(5)) {
        Ok(first)
    } else {
        assert_eq!(first, Event::Opened(WorkerId(5)));
        rx.recv_timeout(Duration::from_secs(1))
    };
    assert_eq!(reaped.unwrap(), Event::Exited(WorkerId(5)), "direct worker exit must not depend on stdout EOF");
    assert!(launcher.owned_workers().is_empty());
}

#[test]
fn the_worker_gets_its_arguments() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("args");
    let program = script(dir.path(), &format!("echo \"$@\" > {}", out.display()));
    let (tx, rx) = channel();
    let mut l = ProcessLauncher::new(program, "m.toml".into(), "b.sock".into(), tx);
    start(&mut l, WorkerId(2), "facecam");
    assert_eq!(rx.recv_timeout(Duration::from_secs(10)).unwrap(), Event::Exited(WorkerId(2)));
    let args = std::fs::read_to_string(out).unwrap();
    assert_eq!(args.trim(), "--manifest m.toml --device facecam --node /dev/null --socket b.sock");
}

#[test]
fn a_restarted_worker_keeps_the_supervisors_fixed_bandwidth_share() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("args");
    let program = script(dir.path(), &format!("echo \"$@\" > {}", out.display()));
    let (tx, rx) = channel();
    let mut launcher =
        ProcessLauncher::new(program, "m.toml".into(), "b.sock".into(), tx).with_mac_raw_share(13_333_333);
    start(&mut launcher, WorkerId(1), "cam");
    assert_eq!(rx.recv_timeout(Duration::from_secs(10)).unwrap(), Event::Exited(WorkerId(1)));
    start(&mut launcher, WorkerId(2), "cam");
    assert_eq!(rx.recv_timeout(Duration::from_secs(10)).unwrap(), Event::Exited(WorkerId(2)));
    assert_eq!(
        std::fs::read_to_string(out).unwrap().trim(),
        "--mac-raw-share 13333333 --manifest m.toml --device cam --node /dev/null --socket b.sock"
    );
}
