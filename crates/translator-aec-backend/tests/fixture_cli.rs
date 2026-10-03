use std::process::Command;

#[test]
fn fixture_rejects_missing_private_pipewire_descriptor() {
    let result = Command::new(env!("CARGO_BIN_EXE_translator-aec-fixture"))
        .output()
        .expect("fixture starts");
    assert!(
        !result.status.success(),
        "fixture must not discover a default server"
    );
}
