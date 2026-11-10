//! A static passthrough capability query precedes all configuration and service initialization.

#[test]
fn capability_query_never_opens_config_socket_or_storage() {
    let directory = tempfile::tempdir().unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_continuum-tier-worker"))
        .current_dir(directory.path())
        .args([
            "--capabilities",
            "--manifest",
            "/nonexistent/private-camera-settings",
            "--socket",
            "/nonexistent/broker.sock",
            "--chunks",
            "never-created",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty(), "static query must not diagnose configuration or start a worker");
    let response: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(response["version"], 1);
    assert_eq!(response["rtsp_h264"], true, "passthrough needs no FFmpeg encoder");
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
}

#[test]
fn rtsp_manifest_devices_follow_their_video_stream() {
    let streams = continuum_tier::service::manifest_streams(
        "[[device]]\nname='network-camera'\nkind='rtsp'\n[[device]]\nname='microphone'\nkind='microphone'\n",
    )
    .unwrap();
    assert_eq!(streams, ["network-camera.video", "microphone.audio"]);
}
