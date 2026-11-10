//! A static capability query must work before configuration, device enumeration, or a broker connection.

#[test]
fn capability_query_precedes_manifest_socket_and_device_access() {
    let directory = tempfile::tempdir().unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_continuum-capture"))
        .current_dir(directory.path())
        .args([
            "--capabilities",
            "--manifest",
            "/nonexistent/private-camera-settings",
            "--socket",
            "/nonexistent/broker.sock",
            "--list",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty(), "static query must not initialize devices or diagnose private config");
    let response: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(response["version"], 1);
    assert_eq!(response["rtsp_h264"], cfg!(feature = "ffmpeg"));
    // rtsps:// also needs a TLS library in the FFmpeg it was built against.
    assert!(response["rtsps"].is_boolean());
    assert!(cfg!(feature = "ffmpeg") || response["rtsps"] == false);
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
}
