use std::{
    process::{Command, Stdio},
    sync::{Arc, Barrier},
    time::{Duration, Instant},
};

#[test]
fn concurrent_panics_emit_only_fixed_private_records() {
    const TEST: &str = "concurrent_panics_emit_only_fixed_private_records";
    const SENTINEL: &str = "synthetic-private-panic-sentinel-62741";
    if std::env::var("TRANSLATOR_PANIC_PRIVACY_CHILD")
        .ok()
        .as_deref()
        == Some(TEST)
    {
        let install = Arc::new(Barrier::new(3));
        let report = Arc::new(Barrier::new(3));
        let workers: Vec<_> = (0..2)
            .map(|_| {
                let install = Arc::clone(&install);
                let report = Arc::clone(&report);
                std::thread::spawn(move || {
                    install.wait();
                    translator_daemon::install_private_panic_hook();
                    report.wait();
                    panic!("{SENTINEL}");
                })
            })
            .collect();
        install.wait();
        translator_daemon::install_private_panic_hook();
        report.wait();
        assert!(std::panic::catch_unwind(|| panic!("{SENTINEL}")).is_err());
        for worker in workers {
            assert!(worker.join().is_err());
        }
        return;
    }
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([TEST, "--exact", "--nocapture"])
        .env("TRANSLATOR_PANIC_PRIVACY_CHILD", TEST)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while child.try_wait().unwrap().is_none() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    let timed_out = child.try_wait().unwrap().is_none();
    if timed_out {
        child.kill().unwrap();
    }
    let output = child.wait_with_output().unwrap();
    assert!(!timed_out && output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        !stderr.contains(SENTINEL),
        "panic payload escaped into stderr"
    );
    assert!(
        !stderr.contains("panic_privacy.rs"),
        "panic location escaped into stderr"
    );
    let records: Vec<_> = stderr.lines().collect();
    assert_eq!(
        records,
        vec![r#"{"event":"runtime_panic","code":"internal_error"}"#; 3]
    );
}
