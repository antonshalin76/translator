#[path = "../src/aec_backend_session.rs"]
#[allow(dead_code)] // Physical native APIs are not launched by CPU custody tests.
mod session;

use std::{
    io::Write,
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use tempfile::NamedTempFile;

use rustix::{
    fd::OwnedFd,
    process::{Pid, PidfdFlags, Signal, kill_process, pidfd_open, pidfd_send_signal},
};
use tokio::process::Command;

use session::{AecBackendSessionError, AecBackendSessionOwner, AecBackendSessionStatus};

fn native_packet(file: &mut NamedTempFile, body: &[u8]) {
    file.write_all(&(body.len() as u32).to_le_bytes()).unwrap();
    file.write_all(body).unwrap();
}

fn native_frame(file: &mut NamedTempFile, sequence: u64, pcm: &[u8]) {
    let mut body = Vec::with_capacity(72 + pcm.len());
    body.extend_from_slice(&[2, 1, 0, 0]);
    body.extend_from_slice(&1_u64.to_le_bytes()); // session
    body.extend_from_slice(&1_u64.to_le_bytes()); // generation
    body.extend_from_slice(&sequence.to_le_bytes());
    body.extend_from_slice(&7_u32.to_le_bytes()); // clock
    body.extend_from_slice(&480_u32.to_le_bytes());
    body.extend_from_slice(&(9_600 + sequence * 480).to_le_bytes());
    body.extend_from_slice(&480_u64.to_le_bytes());
    body.extend_from_slice(&0_u64.to_le_bytes()); // xrun
    body.extend_from_slice(&1_u32.to_le_bytes());
    body.extend_from_slice(&48_000_u32.to_le_bytes());
    body.extend_from_slice(&10_u32.to_le_bytes()); // node
    body.extend_from_slice(pcm);
    native_packet(file, &body);
}

fn native_stream_file() -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();
    let mut hello = vec![1, 1, 0, 0];
    hello.extend_from_slice(&1_u64.to_le_bytes());
    hello.extend_from_slice(&1_u64.to_le_bytes());
    hello.extend_from_slice(&10_u32.to_le_bytes());
    hello.extend_from_slice(&48_000_u32.to_le_bytes());
    hello.extend_from_slice(&480_u32.to_le_bytes());
    hello.extend_from_slice(&480_u32.to_le_bytes());
    native_packet(&mut file, &hello);
    let mut links = vec![6, 1, 0, 0];
    links.extend_from_slice(&1_u64.to_le_bytes());
    links.extend_from_slice(&1_u64.to_le_bytes());
    links.extend_from_slice(&10_u32.to_le_bytes());
    for (local, link, peer) in [(1_u32, 31_u32, 20_u32), (2, 32, 21), (3, 33, 22)] {
        for value in [local, link, peer, 1] {
            links.extend_from_slice(&value.to_le_bytes());
        }
    }
    native_packet(&mut file, &links);
    let mut armed = vec![7, 1, 0, 0];
    armed.extend_from_slice(&1_u64.to_le_bytes());
    armed.extend_from_slice(&1_u64.to_le_bytes());
    armed.extend_from_slice(&10_u32.to_le_bytes());
    native_packet(&mut file, &armed);
    let mut pcm = Vec::with_capacity(480 * 12);
    for _ in 0..480 {
        for value in [0.25_f32, 0.0, 0.125] {
            pcm.extend_from_slice(&value.to_le_bytes());
        }
    }
    for sequence in 0..4_500 {
        native_frame(&mut file, sequence, &pcm);
    }
    file.flush().unwrap();
    file
}

fn copied_stream(source: &NamedTempFile) -> NamedTempFile {
    let copy = NamedTempFile::new().unwrap();
    std::fs::copy(source.path(), copy.path()).unwrap();
    copy
}

async fn run_file(
    path: &std::path::Path,
    exit_code: i32,
) -> (
    AecBackendSessionStatus,
    Option<session::AecBackendTransportResult>,
) {
    let mut command = if exit_code == 0 {
        let mut command = Command::new("/usr/bin/cat");
        command.arg(path);
        command
    } else {
        let mut command = Command::new("/usr/bin/bash");
        command
            .arg("-c")
            .arg("cat -- \"$1\"; exit 7")
            .arg("owner-test")
            .arg(path);
        command
    };
    command.stderr(Stdio::null());
    let owner = AecBackendSessionOwner::new();
    let attempt = owner.start(command).await.unwrap();
    assert!(attempt.take_nonadmissible_result().await.is_none());
    let terminal = tokio::time::timeout(Duration::from_secs(20), attempt.wait())
        .await
        .expect("synthetic native stream exceeded owner deadline");
    assert_eq!(owner.status().await, AecBackendSessionStatus::Idle);
    let result = attempt.take_nonadmissible_result().await;
    assert!(attempt.take_nonadmissible_result().await.is_none());
    (terminal, result)
}

fn stubborn_process() -> Command {
    let mut command = Command::new("/usr/bin/python3");
    command
        .arg("-I")
        .arg("-c")
        .arg(
            "import signal, subprocess, time\n"
                .to_owned()
                + "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                + "subprocess.Popen(['/usr/bin/python3', '-I', '-c', "
                + "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'])\n"
                + "time.sleep(30)\n",
        )
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    command
}

#[tokio::test]
async fn cancel_is_shared_and_reaps_owned_process_group() {
    let owner = AecBackendSessionOwner::new();
    let attempt = owner.start(stubborn_process()).await.unwrap();
    assert_eq!(owner.status().await, AecBackendSessionStatus::Running);
    assert_eq!(owner.verified_frames(), 0);
    assert!(matches!(
        owner.start(stubborn_process()).await,
        Err(AecBackendSessionError::Busy)
    ));
    tokio::time::sleep(Duration::from_millis(100)).await;
    let started = tokio::time::Instant::now();
    let terminal = tokio::time::timeout(Duration::from_secs(9), attempt.cancel())
        .await
        .expect("cancel did not finish within the shared cleanup budget");
    assert_eq!(terminal, AecBackendSessionStatus::Cancelled);
    assert!(started.elapsed() <= Duration::from_secs(8));
    assert_eq!(owner.status().await, AecBackendSessionStatus::Idle);
}

#[tokio::test]
async fn owned_group_is_a_private_session() {
    let owner = AecBackendSessionOwner::new();
    let attempt = owner.start(stubborn_process()).await.unwrap();
    let group = Pid::from_raw(i32::try_from(attempt.owned_group()).unwrap()).unwrap();
    assert_eq!(rustix::process::getsid(Some(group)).unwrap(), group);
    assert!(
        rustix::process::setpgid(None, Some(group)).is_err(),
        "unrelated process joined the owned group"
    );
    assert_eq!(attempt.cancel().await, AecBackendSessionStatus::Cancelled);
    assert_eq!(owner.status().await, AecBackendSessionStatus::Idle);
}

#[tokio::test]
async fn same_session_member_in_other_group_prevents_early_reap() {
    let pid_file = NamedTempFile::new().unwrap();
    let mut command = Command::new("/usr/bin/python3");
    command
        .arg("-I")
        .arg("-c")
        .arg(
            "import subprocess,sys,time\n".to_owned()
                + "child=subprocess.Popen(['/usr/bin/python3','-I','-c',"
                + "'import os,time; os.setpgid(0,0); time.sleep(30)'],"
                + "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
                + "open(sys.argv[1],'w').write(str(child.pid))\n"
                + "time.sleep(30)\n",
        )
        .arg(pid_file.path())
        .stderr(Stdio::null());
    let owner = AecBackendSessionOwner::new();
    let attempt = owner.start(command).await.unwrap();
    let group = Pid::from_raw(i32::try_from(attempt.owned_group()).unwrap()).unwrap();
    let descendant = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if let Ok(value) = std::fs::read_to_string(pid_file.path()) {
                if let Ok(pid) = value.parse::<i32>() {
                    if let Some(pid) = Pid::from_raw(pid) {
                        if rustix::process::getpgid(Some(pid)).is_ok_and(|pgid| pgid != group) {
                            break pid;
                        }
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("descendant did not move to its own group");
    let _supervisor = DescendantSupervisor {
        pidfd: pidfd_open(descendant, PidfdFlags::empty()).unwrap(),
        gate: Arc::new(AtomicBool::new(false)),
    };
    assert_eq!(rustix::process::getsid(Some(descendant)).unwrap(), group);
    assert_eq!(
        attempt.cancel().await,
        AecBackendSessionStatus::CleanupPending
    );
    assert_eq!(
        owner.status().await,
        AecBackendSessionStatus::CleanupPending
    );
    assert!(attempt.take_nonadmissible_result().await.is_none());
    pidfd_send_signal(&_supervisor.pidfd, Signal::KILL).unwrap();
    tokio::time::timeout(Duration::from_secs(10), async {
        while owner.status().await != AecBackendSessionStatus::Idle {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("same-session descendant still held leader custody");
    assert!(!process_is_live(descendant.as_raw_pid() as u32));
}

#[tokio::test]
async fn concurrent_cancel_calls_join_one_cleanup() {
    let owner = AecBackendSessionOwner::new();
    let attempt = owner.start(stubborn_process()).await.unwrap();
    let (first, second) = tokio::time::timeout(Duration::from_secs(9), async {
        tokio::join!(attempt.cancel(), attempt.cancel())
    })
    .await
    .expect("repeat cancel did not join the same bounded cleanup");
    assert_eq!(first, AecBackendSessionStatus::Cancelled);
    assert_eq!(second, AecBackendSessionStatus::Cancelled);
    assert_eq!(owner.status().await, AecBackendSessionStatus::Idle);
}

#[tokio::test]
async fn owner_shutdown_joins_the_same_cleanup_task() {
    let owner = AecBackendSessionOwner::new();
    let attempt = owner.start(stubborn_process()).await.unwrap();
    let group = attempt.owned_group();
    let status = tokio::time::timeout(Duration::from_secs(9), owner.shutdown())
        .await
        .expect("owner shutdown lost the cleanup task");
    assert_eq!(status, AecBackendSessionStatus::Cancelled);
    assert_eq!(attempt.wait().await, AecBackendSessionStatus::Cancelled);
    assert_eq!(owner.status().await, AecBackendSessionStatus::Idle);
    assert!(attempt.take_nonadmissible_result().await.is_none());
    assert!(matches!(
        owner.start(stubborn_process()).await,
        Err(AecBackendSessionError::Busy)
    ));
    let group_pid = Pid::from_raw(i32::try_from(group).unwrap()).unwrap();
    assert!(matches!(
        rustix::process::test_kill_process_group(group_pid),
        Err(rustix::io::Errno::SRCH)
    ));
}

#[tokio::test]
async fn dropped_caller_still_drives_cleanup() {
    let owner = AecBackendSessionOwner::new();
    let attempt = owner.start(stubborn_process()).await.unwrap();
    drop(attempt);
    tokio::time::timeout(Duration::from_secs(9), async {
        loop {
            if owner.status().await == AecBackendSessionStatus::Idle {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("caller drop left an owned process running");
}

#[tokio::test]
async fn no_progress_is_bounded_and_cannot_become_success() {
    let owner = AecBackendSessionOwner::new();
    let attempt = owner.start(stubborn_process()).await.unwrap();
    let terminal = tokio::time::timeout(Duration::from_secs(13), attempt.wait())
        .await
        .expect("no-progress path hung");
    assert_eq!(terminal, AecBackendSessionStatus::NoProgress);
    assert_eq!(owner.status().await, AecBackendSessionStatus::Idle);
}

#[tokio::test]
async fn exited_leader_does_not_release_descendant_group() {
    let mut command = Command::new("/usr/bin/python3");
    command
        .arg("-I")
        .arg("-c")
        .arg(
            "import signal, subprocess\n"
                .to_owned()
                + "subprocess.Popen(['/usr/bin/python3', '-I', '-c', "
                + "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'], "
                + "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n",
        )
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let owner = AecBackendSessionOwner::new();
    let attempt = owner.start(command).await.unwrap();
    let terminal = tokio::time::timeout(Duration::from_secs(9), attempt.wait())
        .await
        .expect("leader exit left its descendant group unowned");
    assert_eq!(terminal, AecBackendSessionStatus::SourceFailed);
    assert_eq!(owner.status().await, AecBackendSessionStatus::Idle);
}

struct DescendantSupervisor {
    pidfd: OwnedFd,
    gate: Arc<AtomicBool>,
}

impl Drop for DescendantSupervisor {
    fn drop(&mut self) {
        self.gate.store(false, Ordering::Release);
        let _ = pidfd_send_signal(&self.pidfd, Signal::KILL);
    }
}

fn process_is_live(pid: u32) -> bool {
    let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
        return false;
    };
    stat.rsplit_once(") ")
        .and_then(|(_, fields)| fields.split_whitespace().next())
        .is_some_and(|state| state != "Z" && state != "X")
}

#[tokio::test]
async fn failed_first_cleanup_keeps_live_descendant_owned_until_retry() {
    let pid_file = NamedTempFile::new().unwrap();
    let gate = Arc::new(AtomicBool::new(true));
    let mut command = Command::new("/usr/bin/python3");
    command
        .arg("-I")
        .arg("-c")
        .arg(
            "import subprocess,sys\n"
                .to_owned()
                + "child=subprocess.Popen(['/usr/bin/python3','-I','-c',"
                + "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(45)'],"
                + "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
                + "open(sys.argv[1],'w').write(str(child.pid))\n",
        )
        .arg(pid_file.path())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let owner = AecBackendSessionOwner::with_cleanup_signal_gate(Arc::clone(&gate));
    let attempt = owner.start(command).await.unwrap();
    let descendant = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if let Ok(value) = std::fs::read_to_string(pid_file.path()) {
                if let Ok(pid) = value.parse::<u32>() {
                    break pid;
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("descendant did not start");
    assert!(process_is_live(descendant));
    let descendant_pid = Pid::from_raw(i32::try_from(descendant).unwrap()).unwrap();
    let _supervisor = DescendantSupervisor {
        pidfd: pidfd_open(descendant_pid, PidfdFlags::empty()).unwrap(),
        gate: Arc::clone(&gate),
    };
    let status = tokio::time::timeout(Duration::from_secs(15), attempt.wait())
        .await
        .expect("first cleanup never returned");
    assert_eq!(status, AecBackendSessionStatus::CleanupPending);
    assert_eq!(
        owner.status().await,
        AecBackendSessionStatus::CleanupPending
    );
    assert!(process_is_live(descendant), "failure barrier did not hold");
    assert!(attempt.take_nonadmissible_result().await.is_none());
    assert!(matches!(
        owner.start(stubborn_process()).await,
        Err(AecBackendSessionError::Busy)
    ));
    let first_shutdown = owner.shutdown().await;
    assert_eq!(first_shutdown, AecBackendSessionStatus::CleanupPending);
    assert!(process_is_live(descendant));
    gate.store(false, Ordering::Release);
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if owner.status().await == AecBackendSessionStatus::Idle {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("same owner lost retry authority");
    assert!(!process_is_live(descendant), "descendant survived retry");
    assert_eq!(owner.shutdown().await, AecBackendSessionStatus::Idle);
    assert!(attempt.take_nonadmissible_result().await.is_none());
}

#[tokio::test]
async fn isolated_runner_rejects_stream_with_foreign_session() {
    let file = native_stream_file();
    let root = tempfile::tempdir().unwrap();
    let scope_path = root.path().join("owned.scope");
    std::fs::create_dir(&scope_path).unwrap();
    std::fs::write(scope_path.join("cgroup.kill"), b"0").unwrap();
    let cleanup_path = scope_path.clone();
    let cleanup_observer = tokio::spawn(async move {
        loop {
            if std::fs::read(cleanup_path.join("cgroup.kill"))
                .ok()
                .as_deref()
                == Some(b"1")
            {
                std::fs::remove_file(cleanup_path.join("cgroup.kill")).unwrap();
                std::fs::remove_dir(cleanup_path).unwrap();
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    });
    let mut command = Command::new("/usr/bin/cat");
    command.arg(file.path()).stderr(Stdio::null());
    let owner = AecBackendSessionOwner::new();
    let attempt = owner
        .start_with_test_scope(command, scope_path.clone(), 2)
        .await
        .unwrap();
    let terminal = tokio::time::timeout(Duration::from_secs(10), attempt.wait())
        .await
        .expect("foreign-session stream exceeded cleanup budget");
    assert_eq!(terminal, AecBackendSessionStatus::Invalidated);
    tokio::time::timeout(Duration::from_secs(2), cleanup_observer)
        .await
        .expect("bound test scope was not killed")
        .unwrap();
    assert!(!scope_path.exists());
    assert_eq!(owner.status().await, AecBackendSessionStatus::Idle);
    assert!(attempt.take_nonadmissible_result().await.is_none());
}

#[tokio::test]
#[ignore = "requires the prebuilt native AEC backend and an isolated PipeWire namespace"]
async fn isolated_native_stream_is_owned_and_cancelled() {
    let owner = AecBackendSessionOwner::new();
    let attempt = owner.start_isolated_runner().await.unwrap();
    let (_, scope_path) = attempt.owned_scope().expect("owned transient scope");
    let scope_path = scope_path.to_path_buf();

    tokio::time::timeout(Duration::from_secs(12), async {
        loop {
            if owner.verified_frames() >= 10 {
                break;
            }
            assert_eq!(
                owner.status().await,
                AecBackendSessionStatus::Running,
                "private graph ended before ten verified frames"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("private graph did not provide ten verified frames");

    assert!(
        scope_path.exists(),
        "private user scope was never established"
    );
    let started = tokio::time::Instant::now();
    let terminal = tokio::time::timeout(Duration::from_secs(9), attempt.cancel())
        .await
        .expect("private graph cancellation exceeded its shared cleanup budget");
    assert_eq!(terminal, AecBackendSessionStatus::Cancelled);
    assert!(started.elapsed() <= Duration::from_secs(8));
    assert_eq!(owner.status().await, AecBackendSessionStatus::Idle);
    assert!(!scope_path.exists(), "owned scope survived Idle");
}

#[tokio::test]
#[ignore = "requires the prebuilt native AEC backend and an isolated PipeWire namespace"]
async fn killed_runner_cannot_leave_scope_or_early_idle() {
    let owner = AecBackendSessionOwner::new();
    let attempt = owner.start_isolated_runner().await.unwrap();
    let (_, scope_path) = attempt.owned_scope().expect("owned transient scope");
    let scope_path = scope_path.to_path_buf();
    tokio::time::timeout(Duration::from_secs(12), async {
        loop {
            if owner.verified_frames() >= 10 {
                break;
            }
            assert_eq!(owner.status().await, AecBackendSessionStatus::Running);
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("private graph never produced verified frames");
    assert!(scope_path.exists(), "private scope was never established");

    let pid = Pid::from_raw(i32::try_from(attempt.owned_group()).unwrap()).unwrap();
    kill_process(pid, Signal::KILL).expect("could not kill owned runner");
    let terminal = tokio::time::timeout(Duration::from_secs(9), attempt.wait())
        .await
        .expect("owner did not resolve killed runner within cleanup budget");
    assert!(matches!(
        terminal,
        AecBackendSessionStatus::SourceFailed | AecBackendSessionStatus::Invalidated
    ));
    assert_eq!(owner.status().await, AecBackendSessionStatus::Idle);
    assert!(!scope_path.exists(), "owned scope survived runner SIGKILL");
    assert!(attempt.take_nonadmissible_result().await.is_none());
}

#[tokio::test]
async fn exact_stream_requires_clean_eof_exit_and_cleanup_before_result() {
    let base = native_stream_file();
    let (terminal, result) = run_file(base.path(), 0).await;
    assert_eq!(terminal, AecBackendSessionStatus::Completed);
    let result = result.expect("verified transport windows absent after cleanup");
    assert_eq!(result.windows.len(), 45);
    assert_eq!(result.windows[0].start_sample, 9_600);
    assert_eq!(result.windows[0].end_sample, 57_600);
    assert_eq!(result.windows[44].end_sample, 2_169_600);
    assert_eq!(result.windows[0].first_sequence, 0);
    assert_eq!(result.windows[44].last_sequence, 4_499);
}

#[tokio::test]
async fn late_fatal_extra_frame_partial_eof_and_nonzero_exit_never_return_result() {
    let base = native_stream_file();
    let mut late_fatal = copied_stream(&base);
    let mut fatal = vec![3, 1, 0, 0];
    fatal.extend_from_slice(&1_u64.to_le_bytes());
    fatal.extend_from_slice(&1_u64.to_le_bytes());
    fatal.extend_from_slice(&1_u32.to_le_bytes());
    fatal.extend_from_slice(&4_500_u64.to_le_bytes());
    native_packet(&mut late_fatal, &fatal);
    late_fatal.flush().unwrap();
    let (terminal, result) = run_file(late_fatal.path(), 0).await;
    assert_eq!(terminal, AecBackendSessionStatus::Invalidated);
    assert!(result.is_none());

    let mut extra = copied_stream(&base);
    let mut pcm = Vec::with_capacity(480 * 12);
    for _ in 0..480 {
        for value in [0.25_f32, 0.0, 0.125] {
            pcm.extend_from_slice(&value.to_le_bytes());
        }
    }
    native_frame(&mut extra, 4_500, &pcm);
    extra.flush().unwrap();
    let (terminal, result) = run_file(extra.path(), 0).await;
    assert_eq!(terminal, AecBackendSessionStatus::Invalidated);
    assert!(result.is_none());

    let partial = copied_stream(&base);
    partial
        .as_file()
        .set_len(partial.as_file().metadata().unwrap().len() - 1)
        .unwrap();
    let (terminal, result) = run_file(partial.path(), 0).await;
    assert_eq!(terminal, AecBackendSessionStatus::Invalidated);
    assert!(result.is_none());

    let (terminal, result) = run_file(base.path(), 7).await;
    assert_eq!(terminal, AecBackendSessionStatus::SourceFailed);
    assert!(result.is_none());

    let mut command = Command::new("/usr/bin/bash");
    command
        .arg("-c")
        .arg("cat -- \"$1\"; exec 1>&-; sleep 30")
        .arg("owner-test")
        .arg(base.path())
        .stderr(Stdio::null());
    let owner = AecBackendSessionOwner::new();
    let attempt = owner.start(command).await.unwrap();
    let terminal = tokio::time::timeout(Duration::from_secs(10), attempt.wait())
        .await
        .expect("post-EOF child hang exceeded cleanup budget");
    assert_eq!(terminal, AecBackendSessionStatus::NoProgress);
    assert_eq!(owner.status().await, AecBackendSessionStatus::Idle);
    assert!(attempt.take_nonadmissible_result().await.is_none());
}

#[tokio::test]
#[ignore = "requires the prebuilt native AEC backend and an isolated PipeWire namespace"]
async fn isolated_native_exact_stream_yields_nonadmissible_result_after_cleanup() {
    let owner = AecBackendSessionOwner::new();
    let attempt = owner.start_isolated_runner().await.unwrap();
    let (_, scope_path) = attempt.owned_scope().expect("owned transient scope");
    let scope_path = scope_path.to_path_buf();
    let terminal = tokio::time::timeout(Duration::from_secs(120), attempt.wait())
        .await
        .expect("isolated native 4500-frame stream exceeded timeout");
    assert_eq!(terminal, AecBackendSessionStatus::Completed);
    assert_eq!(owner.verified_frames(), 4_500);
    assert_eq!(owner.status().await, AecBackendSessionStatus::Idle);
    assert!(
        !scope_path.exists(),
        "owned scope survived completed attempt"
    );
    let result = attempt
        .take_nonadmissible_result()
        .await
        .expect("missing transport result");
    assert_eq!(result.windows.len(), 45);
    assert_eq!(result.windows[0].raw.len(), 48_000);
    assert_eq!(result.windows[44].clean.len(), 48_000);
}
