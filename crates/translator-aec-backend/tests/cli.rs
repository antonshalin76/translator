use std::process::Command;

#[test]
fn rejects_missing_private_pipewire_and_ipc_descriptors() {
    let result = Command::new(env!("CARGO_BIN_EXE_translator-aec-backend"))
        .output()
        .expect("helper starts");
    assert!(
        !result.status.success(),
        "helper must not discover a default server"
    );
}

#[test]
fn rejects_unconnected_descriptors_before_creating_a_graph() {
    let result = Command::new(env!("CARGO_BIN_EXE_translator-aec-backend"))
        .args([
            "--pipewire-fd",
            "9999",
            "--ipc-fd",
            "9998",
            "--session-id",
            "0123456789abcdef",
        ])
        .output()
        .expect("helper starts");
    assert!(
        !result.status.success(),
        "invalid descriptors must fail closed"
    );
}
