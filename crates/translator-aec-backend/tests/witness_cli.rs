use std::process::Command;

#[test]
fn witness_rejects_missing_private_descriptors() {
    let result = Command::new(env!("CARGO_BIN_EXE_translator-aec-witness"))
        .output()
        .expect("witness starts");
    assert!(
        !result.status.success(),
        "witness must not discover default PipeWire or output sockets"
    );
}
