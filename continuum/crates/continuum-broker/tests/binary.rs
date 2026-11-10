//! The `continuum-broker` binary itself: how it fails, and that it logs in the pack's format.

use std::process::Command;

#[test]
fn printing_a_capture_plan_creates_no_state_or_shared_arena() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("devices.toml");
    let state = dir.path().join("state-must-not-exist");
    std::fs::write(
        &manifest,
        "[[device]]\nname='camera'\nkind='camera'\nvideo={width=1920,height=1080,fps=30,format='UYVY'}",
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_continuum-broker"))
        .args(["--print-tier0-plan", "--state-dir"])
        .arg(&state)
        .arg("--manifest")
        .arg(&manifest)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("camera=camera format=UYVY fps=30 ring_seconds=2 slots_per_class=83"));
    assert!(text.contains("total_arena_bytes="));
    assert!(!state.exists());
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1, "only the supplied manifest exists");
}

#[test]
fn printing_explicit_classes_ignores_even_a_corrupt_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("devices.toml");
    std::fs::write(&manifest, "invalid [").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_continuum-broker"))
        .args(["--print-tier0-plan", "--tier0-classes", "16384x4", "--manifest"])
        .arg(manifest)
        .output()
        .unwrap();
    assert!(out.status.success());
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("source=explicit --tier0-classes"));
    assert!(text.contains("configured_slot_bytes=16384 count=4"));
    assert!(!text.contains("camera="));
}

#[test]
fn printing_a_corrupt_manifest_fails_actionably_without_creating_state() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("devices.toml");
    let state = dir.path().join("state-must-not-exist");
    std::fs::write(&manifest, "invalid [").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_continuum-broker"))
        .args(["--print-tier0-plan", "--state-dir"])
        .arg(&state)
        .arg("--manifest")
        .arg(&manifest)
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8(out.stderr).unwrap().contains("invalid capture manifest"));
    assert!(!state.exists());
}

/// Whether `line` is in the pack's log format with this level, as `services.ERROR_LINE` and the
/// Debug tab read it: an ISO stamp with offset (`2026-11-03T02:20:15-08:00`), then `[tag] LEVEL `.
fn is_pack_line(line: &str, level: &str) -> bool {
    const STAMP: usize = 25;
    let b = line.as_bytes();
    let stamp_ok = b.len() > STAMP
        && b[..STAMP].iter().enumerate().all(|(i, c)| match i {
            4 | 7 => *c == b'-',
            10 => *c == b'T',
            13 | 16 | 22 => *c == b':',
            19 => *c == b'-' || *c == b'+',
            _ => c.is_ascii_digit(),
        })
        && b[STAMP] == b' ';
    let Some(rest) = line.get(STAMP + 1..).and_then(|r| r.strip_prefix('[')) else { return false };
    let Some((tag, rest)) = rest.split_once("] ") else { return false };
    stamp_ok && !tag.contains(']') && rest.starts_with(&format!("{level} "))
}

#[test]
fn a_path_that_is_not_a_socket_fails_with_an_error_line_in_the_pack_format() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("broker.sock");
    std::fs::write(&file, b"not a socket").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_continuum-broker")).arg("--socket").arg(&file).output().unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8(out.stderr).unwrap();
    let line = stderr.lines().find(|l| l.contains("not a socket")).expect("the reason is logged");
    assert!(is_pack_line(line, "ERROR"), "{line:?}");
    assert!(line.contains("[continuum-broker] ERROR "), "{line:?}");
    assert!(!stderr.contains('\u{1b}'), "no color codes in the log");
    assert!(out.stdout.is_empty(), "logs go to stderr, not stdout");
}

/// A running broker that's killed if the test ends without stopping it (a panic), so none outlives the test.
struct Running(Option<std::process::Child>);

impl Drop for Running {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Starts the binary on `socket` with a small arena, and waits until it accepts connections.
fn start(socket: &std::path::Path) -> Running {
    let running = Running(Some(
        Command::new(env!("CARGO_BIN_EXE_continuum-broker"))
            .arg("--socket")
            .arg(socket)
            .args(["--tier0-classes", "16384x4", "--tier0-ring-mib", "1"])
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap(),
    ));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while std::os::unix::net::UnixStream::connect(socket).is_err() {
        assert!(std::time::Instant::now() < deadline, "the broker didn't start listening");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    running
}

fn stop(mut running: Running) {
    let pid = running.0.as_ref().expect("running").id();
    // SIGTERM: a clean stop, which removes the socket and unlinks the arena.
    // Signalled while the guard still holds it, so a failure here still kills it; taken out only for the final wait.
    let pid = rustix::process::Pid::from_raw(i32::try_from(pid).unwrap()).unwrap();
    rustix::process::kill_process(pid, rustix::process::Signal::TERM).unwrap();
    let mut child = running.0.take().expect("running");
    assert!(child.wait().unwrap().success());
}

#[test]
fn the_binary_keeps_its_catalog_beside_the_socket_across_restarts() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("continuum").join("broker.sock");
    let catalog = dir.path().join("continuum").join("catalog.redb");
    stop(start(&socket));
    assert!(catalog.exists(), "the redb catalog is created beside the socket");
    let arena = std::fs::read_to_string(continuum_broker::tier0::generation_path(&socket)).unwrap();
    let name = arena.lines().nth(1).unwrap().to_owned();
    assert!(continuum_shm::Arena::open(&name).is_err(), "a clean stop unlinks the arena");
    // A second run opens the same catalog (redb refuses a file another process holds, so the first let go of it).
    stop(start(&socket));
    assert!(!socket.exists(), "the socket is removed on stop");
}

/// redb keeps a "recovery required" bit in its header's god byte (offset 9) while a writer has the file open, and
/// clears it only when the `Database` closes. A file left with it set is repaired on the next open: a full checksum
/// walk of the whole file (a few minutes on a busy machine for a catalog of several GiB), which can keep a large
/// catalog from opening before its startup timeout.
fn redb_recovery_required(catalog: &std::path::Path) -> bool {
    let header = std::fs::read(catalog).unwrap();
    header[9] & 2 != 0
}

#[test]
fn a_clean_stop_closes_the_catalog_so_the_next_open_needs_no_repair() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("continuum").join("broker.sock");
    let catalog = dir.path().join("continuum").join("catalog.redb");
    let running = start(&socket);
    assert!(redb_recovery_required(&catalog), "a running broker holds the catalog open for writing");
    stop(running);
    assert!(!redb_recovery_required(&catalog), "SIGTERM left the catalog as a crash would, for redb to repair");
    // And again after a run that reopened it.
    stop(start(&socket));
    assert!(!redb_recovery_required(&catalog));
}

#[test]
fn notices_the_broker_read_before_a_stop_are_in_the_catalog_after_it() {
    use continuum_catalog::{Catalog, RedbCatalog};
    use continuum_client::{Client, ClientOptions, StreamSpec, StreamStorage};
    use continuum_core::schema::continuum as fb;
    use continuum_core::{HeaderBuf, HeaderSpec, StreamId, Uuid};

    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("continuum").join("broker.sock");
    let catalog = dir.path().join("continuum").join("catalog.redb");
    let running = start(&socket);
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let (client, ids) = rt.block_on(async {
        let client = Client::connect(&socket, ClientOptions::new("producer")).await.unwrap();
        let spec = StreamSpec {
            name: "cam0".into(),
            kind: fb::StreamKind::Sensor,
            format: fb::PayloadFormat::EncodedVideo,
            shape_name: "MJPEG".into(),
            shape_version: 1,
            nominal_rate_hz: 30.0,
            storage: StreamStorage::Slab { class: 0 },
            anchor: None,
            layer: fb::DomainLayer::Layer1Raw,
        };
        let cam = StreamId(client.register_stream(&spec).await.unwrap().stream_id);
        let ids: Vec<Uuid> = (0..200u32).map(|_| Uuid(uuid::Uuid::now_v7().into_bytes())).collect();
        for (i, id) in ids.iter().enumerate() {
            let at = 1_700_000_000_000_000_000 + i as u64 * 33_333_333;
            client.ingest(&HeaderBuf::build(&HeaderSpec::raw(*id, cam, at, at)).unwrap(), None).unwrap();
        }
        // The broker has read every notice (one session reads in order) but needn't have committed them.
        client.heartbeat(None).await.unwrap();
        (client, ids)
    });
    stop(running);
    drop(client);
    let c = RedbCatalog::open(&catalog).unwrap();
    let missing = ids.iter().filter(|id| c.get(**id).unwrap().is_none()).count();
    assert_eq!(missing, 0, "a stop must commit what the sessions handed over before closing the catalog");
}
