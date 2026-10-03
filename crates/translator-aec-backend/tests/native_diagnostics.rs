use std::{env, fs, process::Command};

#[test]
fn native_first_failure_diagnostics_are_immutable() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let flags = Command::new("pkg-config")
        .args(["--cflags", "--libs", "libpipewire-0.3"])
        .output()
        .expect("pkg-config must run");
    assert!(
        flags.status.success(),
        "PipeWire development flags required"
    );
    let flags = String::from_utf8(flags.stdout).expect("UTF-8 flags");
    let binary = env::temp_dir().join(format!(
        "translator-aec-native-diagnostics-{}",
        std::process::id()
    ));
    let compiled = Command::new("cc")
        .current_dir(manifest)
        .args([
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-DPW_ID_INVALID=SPA_ID_INVALID",
            "-DTRANSLATOR_AEC_LIBRARY=\"unused-in-test\"",
            "tests/native_diagnostics.c",
            "-o",
        ])
        .arg(&binary)
        .args(flags.split_whitespace())
        .args(["-ldl", "-pthread", "-lm"])
        .output()
        .expect("C compiler must run");
    assert!(
        compiled.status.success(),
        "native test compile failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let run = Command::new(&binary)
        .output()
        .expect("native test must run");
    let _ = fs::remove_file(binary);
    assert!(
        run.status.success(),
        "native diagnostic assertion failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
}
