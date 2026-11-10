//! Capability inspection must never initialize a broker or touch configured paths.
#[test]
fn capabilities_early_exit_ignores_configuration_and_leaves_no_state() {
    let dir = tempfile::tempdir().unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_continuum-broker"))
        .current_dir(dir.path())
        .args(["--capabilities", "--manifest", "missing/private.toml", "--socket", "missing/broker.sock"])
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["rtsp_h264"], true);
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
}
