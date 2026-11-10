//! CLI control requests use only temporary local sockets and a fake supervisor; no capture hardware is opened.
use std::process::Command;
use std::time::Duration;

use continuum_capture::control::{ControlServer, Reply};

#[test]
fn the_cli_reports_unavailable_and_failed_with_an_exact_manifest_identity() {
    let dir = tempfile::tempdir().unwrap();
    let invoke = || {
        Command::new(env!("CARGO_BIN_EXE_continuum-capture"))
            .args(["--restart-camera", "cam0", "--state-dir"])
            .arg(dir.path())
            .output()
            .unwrap()
    };
    let absent = invoke();
    assert_eq!(absent.status.code(), Some(3));
    let reply: Reply = serde_json::from_slice(&absent.stdout).unwrap();
    assert_eq!((reply.status.as_str(), reply.device.as_str()), ("unavailable", "cam0"));
    let (_server, requests) = ControlServer::bind(&dir.path().join("continuum/capture.sock")).unwrap();
    // The wire server's fake supervisor returns a reply through a real driver in unit tests. The integration
    // client below intentionally gets a disconnected supervisor, which must be failed (never fallback eligible).
    drop(requests);
    let rejected = invoke();
    assert_eq!(rejected.status.code(), Some(1));
    let reply: Reply = serde_json::from_slice(&rejected.stdout).unwrap();
    assert_eq!((reply.status.as_str(), reply.device.as_str()), ("failed", "cam0"));
}

#[test]
fn an_incompatible_reply_is_failed_even_though_a_control_socket_exists() {
    use std::io::{BufRead, BufReader, Write};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("capture.sock");
    let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
    let peer = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let mut request = String::new();
        BufReader::new(&mut stream).read_line(&mut request).unwrap();
        assert!(request.contains("cam0"));
        stream
            .write_all(
                b"{\"version\":1,\"status\":\"restarting\",\"device\":\"wrong-camera\",\"message\":\"scheduled\"}\n",
            )
            .unwrap();
    });
    let output = Command::new(env!("CARGO_BIN_EXE_continuum-capture"))
        .args(["--restart-camera", "cam0", "--control-socket"])
        .arg(path)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let reply: Reply = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!((reply.status.as_str(), reply.device.as_str()), ("failed", "cam0"));
    peer.join().unwrap();
}

#[test]
fn fable_regression_a_flag_shaped_camera_identity_is_a_literal_value() {
    use std::io::{BufRead, BufReader, Write};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("capture.sock");
    let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
    listener.set_nonblocking(true).unwrap();
    let peer = std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        let (mut stream, _) = loop {
            if let Ok(accepted) = listener.accept() {
                break accepted;
            }
            if std::time::Instant::now() > deadline {
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        let mut request = String::new();
        BufReader::new(&mut stream).read_line(&mut request).unwrap();
        let request: serde_json::Value = serde_json::from_str(&request).unwrap();
        assert_eq!(request["device"], "--camera");
        stream
            .write_all(b"{\"version\":1,\"status\":\"restarting\",\"device\":\"--camera\",\"message\":\"scheduled\"}\n")
            .unwrap();
    });
    let output = Command::new(env!("CARGO_BIN_EXE_continuum-capture"))
        .args(["--restart-camera", "--camera", "--control-socket"])
        .arg(path)
        .output()
        .unwrap();
    peer.join().unwrap();
    assert_eq!(output.status.code(), Some(0), "{}", String::from_utf8_lossy(&output.stderr));
    let reply: Reply = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!((reply.status.as_str(), reply.device.as_str()), ("restarting", "--camera"));
}
