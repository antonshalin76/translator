use std::{
    fs,
    os::unix::process::CommandExt,
    path::Path,
    process::Command,
    thread,
    time::{Duration, Instant},
};

use rustix::process::{Pid, Signal, WaitOptions, kill_process, kill_process_group, waitpid};
use tempfile::tempdir;
use translator_audio::{
    CommandResult, CommandRunError, CommandRunner, ProcessIdentity, SystemCommandRunner,
};

#[test]
fn expired_command_deadline_has_no_observable_effect() {
    let directory = tempdir().unwrap();
    let marker = directory.path().join("spawned");
    let result = SystemCommandRunner.run_until(
        "sh",
        &[
            "-c".to_owned(),
            format!("printf spawned > '{}'", marker.display()),
        ],
        Instant::now(),
    );
    assert!(!marker.exists(), "expired command wrote a marker");
    assert_eq!(result, Err(CommandRunError::DeadlineExpired));
}

#[test]
fn original_command_deadline_wins_over_the_local_limit() {
    run_isolated_probe("short");
}

#[test]
fn system_runner_kills_and_reaps_a_timed_out_child() {
    run_isolated_probe("local");
}

#[test]
fn repeated_command_timeouts_release_direct_child_resources() {
    run_isolated_probe("repeat");
}

#[test]
fn sleeping_command_child() {
    let Some(path) = std::env::var_os("TRANSLATOR_TEST_SLEEP_MARKER") else {
        return;
    };
    let identity = ProcessIdentity::inspect(std::process::id()).unwrap();
    fs::write(
        path,
        serde_json::to_vec(&[
            u64::from(identity.pid),
            identity.start_time_ticks,
            identity.executable_device,
            identity.executable_inode,
        ])
        .unwrap(),
    )
    .unwrap();
    thread::sleep(Duration::from_secs(30));
}

fn cleanup_recorded_child(identity: ProcessIdentity) -> bool {
    let pid = Pid::from_raw(identity.pid as i32).unwrap();
    if ProcessIdentity::inspect(identity.pid) == Some(identity) {
        let _ = kill_process(pid, Signal::KILL);
    }
    // The SUT may have left a zombie: inspect(exe) cannot resolve a zombie, but
    // waitpid can only reap this process's child, never an unrelated reused PID.
    let _ = waitpid(Some(pid), WaitOptions::NOHANG);
    let until = Instant::now() + Duration::from_millis(500);
    while Path::new(&format!("/proc/{}", identity.pid)).exists() && Instant::now() < until {
        let _ = waitpid(Some(pid), WaitOptions::NOHANG);
        thread::sleep(Duration::from_millis(1));
    }
    !Path::new(&format!("/proc/{}", identity.pid)).exists()
}

fn active_command_reader_threads() -> usize {
    fs::read_dir("/proc/self/task")
        .unwrap()
        .map(|entry| entry.unwrap().path().join("comm"))
        .filter(|path| {
            fs::read_to_string(path).is_ok_and(|name| name.trim_end() == "tr-cmd-reader")
        })
        .count()
}

fn command_readers_after_kernel_retirement() -> usize {
    let deadline = Instant::now() + Duration::from_millis(100);
    while active_command_reader_threads() != 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    active_command_reader_threads()
}

#[test]
fn command_timeout_probe_child() {
    let Some(directory) = std::env::var_os("TRANSLATOR_TEST_COMMAND_DIRECTORY") else {
        return;
    };
    let directory = Path::new(&directory);
    let case = std::env::var("TRANSLATOR_TEST_COMMAND_CASE").unwrap();
    SystemCommandRunner.run("true", &[]).unwrap();
    let baseline_fds = fs::read_dir("/proc/self/fd").unwrap().count();
    assert_eq!(command_readers_after_kernel_retirement(), 0);
    let iterations = if case == "repeat" { 5 } else { 1 };
    for iteration in 0..iterations {
        let marker = directory.join(format!("child-{iteration}.json"));
        let script = format!(
            "TRANSLATOR_TEST_SLEEP_MARKER='{}' exec '{}' --exact sleeping_command_child --test-threads=1",
            marker.display(),
            std::env::current_exe().unwrap().display(),
        );
        let started = Instant::now();
        let budget = if case == "local" {
            Duration::from_secs(8)
        } else {
            Duration::from_millis(500)
        };
        let result =
            SystemCommandRunner.run_until("sh", &["-c".to_owned(), script], started + budget);
        let elapsed = started.elapsed();
        let identity = fs::read(&marker)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<[u64; 4]>(&bytes).ok())
            .and_then(|parts| {
                Some(ProcessIdentity {
                    pid: u32::try_from(parts[0]).ok()?,
                    start_time_ticks: parts[1],
                    executable_device: parts[2],
                    executable_inode: parts[3],
                })
            });
        let reaped_before_cleanup =
            identity.is_some_and(|value| !Path::new(&format!("/proc/{}", value.pid)).exists());
        let observed_fds = fs::read_dir("/proc/self/fd").unwrap().count();
        let observed_reader_threads = command_readers_after_kernel_retirement();
        let cleaned = identity.is_some_and(cleanup_recorded_child);

        assert!(cleaned, "fixture child identity missing or cleanup failed");
        assert!(
            reaped_before_cleanup,
            "SUT left its direct child alive or zombie"
        );
        let expected = if case == "local" {
            CommandRunError::TimedOut
        } else {
            CommandRunError::DeadlineExpired
        };
        assert_eq!(result, Err(expected));
        assert!(
            elapsed
                < if case == "local" {
                    Duration::from_secs(4)
                } else {
                    Duration::from_secs(2)
                }
        );
        assert_eq!(
            observed_fds, baseline_fds,
            "reader FDs survived returned command"
        );
        assert_eq!(
            observed_reader_threads, 0,
            "command reader threads survived returned command"
        );
    }
    fs::write(directory.join("completed"), b"all assertions passed").unwrap();
}

fn run_isolated_probe(case: &str) {
    let directory = tempdir().unwrap();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "command_timeout_probe_child",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("TRANSLATOR_TEST_COMMAND_DIRECTORY", directory.path())
        .env("TRANSLATOR_TEST_COMMAND_CASE", case)
        .process_group(0)
        .spawn()
        .unwrap();
    let pid = Pid::from_raw(child.id() as i32).unwrap();
    let until = Instant::now() + Duration::from_secs(10);
    while !directory.path().join("completed").exists() && Instant::now() < until {
        thread::sleep(Duration::from_millis(10));
    }
    let completed = directory.path().join("completed").exists();
    // Child has not been waited on: its PID cannot be reused before this exact
    // isolated group is stopped, including any child missing its marker file.
    let stopped = kill_process_group(pid, Signal::KILL);
    let reaped = child.wait();
    assert!(stopped.is_ok() || stopped == Err(rustix::io::Errno::SRCH));
    assert!(reaped.is_ok());
    assert!(
        completed,
        "isolated command probe failed or exceeded watchdog bound"
    );
}

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
