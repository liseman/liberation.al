//! The capture supervisor: reads the device manifest, watches hotplug (on Linux, cameras from the kernel's uevents
//! and sysfs and microphones from PipeWire's registry; on macOS, AVFoundation's and CoreAudio's device lists), and
//! runs one `continuum-capture-worker` per device, one open at a time (see `continuum_capture::supervisor`).
//!
//! ```sh
//! continuum-capture --manifest config/devices.toml --state-dir data/state   # the broker at data/state/continuum/broker.sock
//! continuum-capture --manifest config/devices.toml --socket /tmp/cnt/broker.sock
//! continuum-capture --restart-camera facecam --state-dir data/state   # accepted, not proof of reopening
//! continuum-capture --list --json [--format MJPEG]  # read-only Linux camera modes and exact intervals
//! continuum-capture --list    # the devices hotplug sees, to write the manifest's match rules from
//! ```

#[path = "../listing.rs"]
mod listing;

use continuum_capture::eprintln;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::mpsc::channel;
use std::time::Duration;

use continuum_capture::driver::{Driver, ProcessLauncher};
use continuum_capture::manifest::Manifest;
use continuum_capture::supervisor::{Event, Supervisor};
use continuum_capture::worker::{STOP, install_stop_signals};

fn main() -> ExitCode {
    continuum_capture::log::stamp_lines();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--capabilities") {
        println!(
            "{}",
            serde_json::json!({"version": 1, "rtsp_h264": cfg!(feature = "ffmpeg"), "device_enabled": true, "device_paused": true})
        );
        return ExitCode::SUCCESS;
    }
    if args.iter().any(|arg| arg == "--restart-camera") {
        return restart_camera(&args);
    }
    if args.iter().any(|arg| arg == "--list") {
        return match listing::Options::parse(&args) {
            Ok(options) => list_devices(&options),
            Err(error) => {
                eprintln!(
                    "continuum-capture: {error}\nusage: continuum-capture --list [--json [--format MJPEG|FOURCC]]"
                );
                ExitCode::from(2)
            }
        };
    }
    let (mut manifest_path, mut socket, mut state_dir, mut control_socket) =
        (None, None, None::<PathBuf>, None::<PathBuf>);
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        match (flag.as_str(), it.next()) {
            ("--manifest", Some(v)) => manifest_path = Some(PathBuf::from(v)),
            ("--socket", Some(v)) => socket = Some(PathBuf::from(v)),
            ("--state-dir", Some(v)) => state_dir = Some(PathBuf::from(v)),
            ("--control-socket", Some(v)) => control_socket = Some(PathBuf::from(v)),
            _ => {
                eprintln!("usage: continuum-capture --manifest P (--state-dir D | --socket S)");
                return ExitCode::from(2);
            }
        }
    }
    let socket = socket.or_else(|| state_dir.map(|d| d.join("continuum").join("broker.sock")));
    let (Some(manifest_path), Some(socket)) = (manifest_path, socket) else {
        eprintln!("usage: continuum-capture --manifest P (--state-dir D | --socket S)");
        return ExitCode::from(2);
    };
    let manifest = match std::fs::read_to_string(&manifest_path)
        .map_err(|e| e.to_string())
        .and_then(|t| Manifest::parse(&t).map_err(|e| e.to_string()))
    {
        Ok(m) => m,
        Err(e) => {
            eprintln!("continuum-capture: {}: {e}", manifest_path.display());
            return ExitCode::from(2);
        }
    };
    #[cfg(target_os = "macos")]
    let mac_raw_share = match continuum_capture::mac_bandwidth::manifest_share(&manifest) {
        Ok(share) => share,
        Err(error) => {
            eprintln!("continuum-capture: macOS bandwidth admission: {error}");
            return ExitCode::from(2);
        }
    };
    let program = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("continuum-capture-worker")))
        .unwrap_or_else(|| PathBuf::from("continuum-capture-worker"));
    let control_socket = control_socket.unwrap_or_else(|| socket.with_file_name("capture.sock"));
    let (_control_server, requests) = match continuum_capture::control::ControlServer::bind(&control_socket) {
        Ok(server) => server,
        Err(error) => {
            eprintln!("continuum-capture: control {}: {error}", control_socket.display());
            return ExitCode::from(2);
        }
    };
    install_stop_signals();
    let (tx, rx) = channel::<Event>();
    for info in continuum_capture::rtsp::configured_devices(&manifest) {
        let _ = tx.send(Event::Added(info));
    }
    watch_hotplug(&tx);
    let grace = Duration::from_millis(manifest.supervisor.stop_grace_ms);
    let launcher = ProcessLauncher::new(program, manifest_path.clone(), socket, tx.clone());
    #[cfg(target_os = "macos")]
    let launcher = launcher.with_mac_raw_share(mac_raw_share);
    let mut driver = Driver::new(Supervisor::new(manifest), launcher, tx)
        .with_pause_status(control_socket.with_file_name("capture_paused.json"));
    driver.run_controlled(&rx, &requests, &manifest_path, &STOP, grace);
    ExitCode::SUCCESS
}

fn restart_camera(args: &[String]) -> ExitCode {
    let mut device = None;
    let mut state_dir = None;
    let mut broker = None;
    let mut control = None;
    let mut seen = std::collections::BTreeSet::new();
    let mut args = args.iter();
    while let Some(flag) = args.next() {
        let Some(value) = args.next() else {
            eprintln!(
                "usage: continuum-capture --restart-camera NAME (--state-dir D | --socket S) [--control-socket C]"
            );
            return ExitCode::from(2);
        };
        if !seen.insert(flag) {
            eprintln!("continuum-capture: duplicate argument {flag}");
            return ExitCode::from(2);
        }
        match flag.as_str() {
            "--restart-camera" => device = Some(value.as_str()),
            "--state-dir" => state_dir = Some(PathBuf::from(value)),
            "--socket" => broker = Some(PathBuf::from(value)),
            "--control-socket" => control = Some(PathBuf::from(value)),
            _ => {
                eprintln!("continuum-capture: unknown argument {flag}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(device) = device.filter(|name| !name.is_empty()) else { return ExitCode::from(2) };
    let control = control
        .or_else(|| broker.map(|path| path.with_file_name("capture.sock")))
        .or_else(|| state_dir.map(|path| path.join("continuum/capture.sock")));
    let Some(control) = control else {
        eprintln!("continuum-capture: --state-dir, --socket, or --control-socket is required");
        return ExitCode::from(2);
    };
    let reply = continuum_capture::control::restart_camera(&control, device);
    let code = match reply.status.as_str() {
        "restarting" => 0,
        "unavailable" => 3,
        _ => 1,
    };
    match serde_json::to_string(&reply) {
        Ok(json) => println!("{json}"),
        Err(error) => {
            eprintln!("continuum-capture: encoding control reply: {error}");
            return ExitCode::FAILURE;
        }
    }
    ExitCode::from(code)
}

#[cfg(target_os = "linux")]
fn watch_hotplug(tx: &std::sync::mpsc::Sender<Event>) {
    use std::collections::BTreeSet;

    use continuum_capture::hotplug::{Roots, UeventSocket, parse_uevent};

    /// What the reader hands the describer: a datagram, or word that the kernel dropped some (the socket overran).
    enum Raw {
        Datagram(Vec<u8>),
        Overrun,
    }

    let roots = Roots::default();
    // The cameras there are now, against those reported: what changed, as events. Used for the first scan and after
    // an overrun, when events were lost.
    let resync = |roots: &Roots, reported: &mut BTreeSet<String>, tx: &std::sync::mpsc::Sender<Event>| -> bool {
        let now = roots.cameras();
        let keys: BTreeSet<String> = now.iter().map(|c| c.key.clone()).collect();
        for gone in reported.difference(&keys).cloned().collect::<Vec<_>>() {
            reported.remove(&gone);
            if tx.send(Event::Removed(gone)).is_err() {
                return false;
            }
        }
        for camera in now {
            // A repeated add is harmless: the supervisor already has it.
            reported.insert(camera.key.clone());
            if tx.send(Event::Added(camera)).is_err() {
                return false;
            }
        }
        true
    };
    // Listen before scanning, so a camera plugged in between isn't missed (a duplicate add is harmless).
    let socket = match UeventSocket::open() {
        Ok(s) => Some(s),
        Err(e) => {
            eprintln!("continuum-capture: no uevent socket, so no camera hotplug: {e}");
            None
        }
    };
    let mut reported = BTreeSet::new();
    resync(&roots, &mut reported, tx);
    if let Err(e) = continuum_capture::pipewire::watch_sources(tx.clone()) {
        eprintln!("continuum-capture: no PipeWire watcher, so no microphones: {e}");
    }
    let Some(socket) = socket else { return };
    // The reader only drains the socket, so the kernel's buffer never fills while the describer waits for udev; the
    // channel between them is unbounded, so nothing is dropped there either.
    let (raw_tx, raw_rx) = std::sync::mpsc::channel::<Raw>();
    let reader = std::thread::Builder::new().name("uevents".into()).spawn(move || {
        loop {
            let item = match socket.recv(1_000) {
                Ok(Some(d)) => Raw::Datagram(d),
                Ok(None) => continue,
                // The kernel dropped uevents because the buffer was full: the describer rescans.
                Err(e) if e.raw_os_error() == Some(libc::ENOBUFS) => Raw::Overrun,
                Err(e) => {
                    eprintln!("continuum-capture: uevents: {e}");
                    return;
                }
            };
            if raw_tx.send(item).is_err() {
                return;
            }
        }
    });
    if let Err(e) = reader {
        eprintln!("continuum-capture: no uevent thread: {e}");
        return;
    }
    let tx = tx.clone();
    let describer = std::thread::Builder::new().name("uevent-describe".into()).spawn(move || {
        // An added camera waits briefly for udev's stable links, which a manifest may match.
        let (settle, step) = (Duration::from_secs(2), Duration::from_millis(100));
        for item in raw_rx {
            let event = match item {
                Raw::Overrun => {
                    eprintln!("continuum-capture: uevents were dropped (the socket overran): rescanning the cameras");
                    if resync(&roots, &mut reported, &tx) {
                        continue;
                    }
                    return;
                }
                Raw::Datagram(d) => parse_uevent(&d).and_then(|u| roots.event_settled(&u, settle, step)),
            };
            let Some(event) = event else { continue };
            match &event {
                Event::Added(c) => {
                    reported.insert(c.key.clone());
                }
                Event::Removed(key) => {
                    reported.remove(key);
                }
                _ => {}
            }
            if tx.send(event).is_err() {
                return;
            }
        }
    });
    if let Err(e) = describer {
        eprintln!("continuum-capture: no uevent thread: {e}");
    }
}

#[cfg(target_os = "macos")]
fn watch_hotplug(tx: &std::sync::mpsc::Sender<Event>) {
    use continuum_capture::devlist::{DeviceList, POLL_MS};

    let tx = tx.clone();
    let spawned = std::thread::Builder::new().name("device lists".into()).spawn(move || {
        let (mut cameras, mut microphones) = (DeviceList::new(), DeviceList::new());
        loop {
            // Each read drains its own autorelease pool (a thread with none would keep what AVFoundation and
            // CoreAudio autorelease, every second, for good).
            let (now_cameras, now_mics) = objc2::rc::autoreleasepool(|_| {
                (continuum_capture::avfoundation::cameras(), continuum_capture::coreaudio::input_devices())
            });
            let events = cameras.update(now_cameras).into_iter().chain(microphones.update(now_mics));
            for e in events {
                if tx.send(e).is_err() {
                    return;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(POLL_MS));
        }
    });
    if let Err(e) = spawned {
        eprintln!("continuum-capture: no device list thread, so no hotplug: {e}");
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn watch_hotplug(_: &std::sync::mpsc::Sender<Event>) {
    eprintln!("continuum-capture: no hotplug on this OS");
}

/// Prints every device hotplug sees now, one a line: what a manifest's match rules can name.
fn list_devices(options: &listing::Options) -> ExitCode {
    let mut devices = Vec::new();
    #[cfg(target_os = "linux")]
    let mut diagnostics = Vec::new();
    #[cfg(not(target_os = "linux"))]
    let diagnostics = Vec::new();
    let (tx, rx) = channel::<Event>();
    #[cfg(target_os = "linux")]
    {
        for camera in continuum_capture::hotplug::Roots::default().cameras() {
            let _ = tx.send(Event::Added(camera));
        }
        // PipeWire's registry reports what it has right away; a second is plenty.
        if let Err(e) = continuum_capture::pipewire::watch_sources(tx.clone()) {
            diagnostics.push(format!("no PipeWire, so no microphones listed: {e}"));
        }
    }
    #[cfg(target_os = "macos")]
    for d in objc2::rc::autoreleasepool(|_| {
        let mut all = continuum_capture::avfoundation::cameras();
        all.extend(continuum_capture::coreaudio::input_devices());
        all
    }) {
        let _ = tx.send(Event::Added(d));
    }
    drop(tx);
    let until = std::time::Instant::now() + Duration::from_secs(1);
    while let Some(left) = until.checked_duration_since(std::time::Instant::now()) {
        match rx.recv_timeout(left) {
            Ok(Event::Added(d)) => {
                devices.push(d);
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }
    if options.json {
        match listing::json(&devices, diagnostics, options) {
            Ok(json) => println!("{json}"),
            Err(error) => {
                eprintln!("continuum-capture: serializing devices: {error}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        for diagnostic in diagnostics {
            eprintln!("continuum-capture: {diagnostic}");
        }
        for d in devices {
            println!("{}", listing::text(&d));
        }
    }
    ExitCode::SUCCESS
}
