use std::{
    fs,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

use tempfile::tempdir;
use translator_audio::{CommandResult, CommandRunError, CommandRunner, SystemCommandRunner};

#[test]
fn system_runner_captures_stdout_and_stderr() {
    let result = SystemCommandRunner
        .run(
            "sh",
            &[
                "-c".to_owned(),
                "printf 'captured-out'; printf 'captured-err' >&2; exit 7".to_owned(),
            ],
        )
        .unwrap();

    assert_eq!(
        result,
        CommandResult::failure(b"captured-out".to_vec(), b"captured-err".to_vec())
    );
}

#[test]
fn system_runner_kills_and_reaps_a_timed_out_child() {
    let directory = tempdir().unwrap();
    let pid_path = directory.path().join("child.pid");
    let script = format!(
        "printf '%s' \"$$\" > '{}'; exec sleep 30",
        pid_path.display()
    );
    let started = Instant::now();

    let error = SystemCommandRunner
        .run("sh", &["-c".to_owned(), script])
        .unwrap_err();

    assert_eq!(error, CommandRunError::TimedOut);
    assert!(
        started.elapsed() <= Duration::from_millis(2_500),
        "command timeout exceeded its hard bound: {:?}",
        started.elapsed()
    );
    let pid = fs::read_to_string(pid_path).unwrap();
    let status = Command::new("kill")
        .args(["-0", pid.trim()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(!status.success(), "timed-out child was not reaped");
}
