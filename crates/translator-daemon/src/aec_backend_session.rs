//! Lifetime owner for isolated AEC transport and retained native sessions.

use std::{
    fs::File,
    os::{
        fd::{AsFd, AsRawFd},
        unix::{fs::MetadataExt, process::CommandExt},
    },
    panic::AssertUnwindSafe,
    path::PathBuf,
    process::{ExitStatus, Stdio},
    sync::{
        Arc, Mutex as StdMutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::Duration,
};

use rustix::{
    fs::{Mode, OFlags, openat},
    io::{Errno, FdFlags, fcntl_setfd},
    pipe::{PipeFlags, pipe_with},
    process::{
        Pid, Signal, WaitId, WaitIdOptions, kill_process_group, test_kill_process_group, waitid,
    },
};
use tokio::{
    io::AsyncReadExt,
    process::{Child, Command},
    sync::{Mutex, watch},
    time::Instant,
};
use translator_audio::{
    AecBackendLink, AecBackendWindow, AecBackendWire, AecGraphIdentity, NativeAecEvent,
    NativeAecHandle, NativeAecLaunchAuthority, NativeAecSource, NativeCaptureReceiver,
    NativeMeasurementReceiver, spawn_retained_native_aec,
};
#[cfg(not(test))]
use uuid::Uuid;

const NO_PROGRESS_BUDGET: Duration = Duration::from_secs(4);
// Reserve the entire cleanup budget inside the 180-second attempt bound.
const RUN_BUDGET: Duration = Duration::from_secs(172);
const COOPERATIVE_CLEANUP: Duration = Duration::from_secs(2);
const TOTAL_CLEANUP: Duration = Duration::from_secs(8);
const POLL_INTERVAL: Duration = Duration::from_millis(20);
const TARGET_FRAMES: u64 = 4_500;
const TARGET_WINDOWS: usize = 45;
const MANAGER_UNKNOWN: u8 = 0;
const MANAGER_REQUEST_SENT: u8 = 4;
const MANAGER_ADMITTED: u8 = 1;
const MANAGER_REJECTED: u8 = 2;
const MANAGER_INVALID: u8 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AecBackendSessionStatus {
    Idle,
    Running,
    Completed,
    CleanupPending,
    Cancelled,
    NoProgress,
    TimedOut,
    Invalidated,
    SourceFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AecBackendSessionError {
    Busy,
    SpawnFailed,
    RandomnessUnavailable,
}

/// Native transport samples from an isolated synthetic graph, not an AEC proof.
pub struct AecBackendTransportResult {
    #[allow(dead_code)] // Exercised by the isolated diagnostic guardian harness.
    pub windows: Vec<AecBackendWindow>,
}

/// Only data/control endpoints leave the process guardian. The source stays owned.
pub(crate) struct RetainedNativeAecAttempt {
    pub guard: AecBackendAttempt,
    pub handle: NativeAecHandle,
    pub measurement: NativeMeasurementReceiver,
    pub capture: NativeCaptureReceiver,
}

struct NativeEndpoints {
    handle: NativeAecHandle,
    measurement: NativeMeasurementReceiver,
    capture: NativeCaptureReceiver,
}

struct StartedBackend {
    group: u32,
    native: Option<NativeEndpoints>,
}

enum BackendLaunch {
    Transport(Box<Command>),
    Native { lifecycle_write: Arc<File> },
}

struct PendingStart(Option<watch::Sender<bool>>);

impl Drop for PendingStart {
    fn drop(&mut self) {
        if let Some(cancel) = &self.0 {
            cancel.send_replace(true);
        }
    }
}

struct OwnerState {
    status: AecBackendSessionStatus,
}

struct OwnerControl {
    task: Option<thread::JoinHandle<()>>,
    cancel: Option<watch::Sender<bool>>,
    terminal: Option<watch::Receiver<Option<AecBackendSessionStatus>>>,
    closing: bool,
}

#[derive(Clone)]
struct AttemptShared {
    state: Arc<Mutex<OwnerState>>,
    verified_frames: Arc<AtomicU64>,
    cleanup_signal_gate: Option<Arc<AtomicBool>>,
    #[cfg(test)]
    collector_panic_gate: Option<Arc<AtomicBool>>,
    terminal_tx: watch::Sender<Option<AecBackendSessionStatus>>,
    result: Arc<Mutex<Option<AecBackendTransportResult>>>,
}

#[derive(Clone)]
struct OwnedScope {
    unit: String,
    cgroup: PathBuf,
    expected_session: u64,
    bound: Arc<StdMutex<Option<File>>>,
    lifecycle_read: Option<Arc<File>>,
    manager_state: Arc<std::sync::atomic::AtomicU8>,
    #[cfg(test)]
    cleanup_negative_observed: Option<Arc<AtomicBool>>,
}

impl OwnedScope {
    fn new() -> Result<Self, AecBackendSessionError> {
        let mut session_bytes = [0_u8; 8];
        getrandom::fill(&mut session_bytes)
            .map_err(|_| AecBackendSessionError::RandomnessUnavailable)?;
        let expected_session = u64::from_le_bytes(session_bytes);
        if expected_session == 0 {
            return Err(AecBackendSessionError::RandomnessUnavailable);
        }
        let uid = rustix::process::getuid().as_raw();
        #[cfg(test)]
        let unit = {
            let value = std::env::var("TRANSLATOR_AEC_NATIVE_SCOPE_UNIT")
                .map_err(|_| AecBackendSessionError::SpawnFailed)?;
            let suffix = value
                .strip_prefix("translator-aec-")
                .ok_or(AecBackendSessionError::SpawnFailed)?;
            if suffix.len() != 32
                || !suffix
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err(AecBackendSessionError::SpawnFailed);
            }
            value
        };
        #[cfg(not(test))]
        let unit = format!("translator-aec-{}", Uuid::new_v4().simple());
        let cgroup = PathBuf::from(format!(
            "/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{unit}.scope"
        ));
        Ok(Self {
            unit,
            cgroup,
            expected_session,
            bound: Arc::new(StdMutex::new(None)),
            lifecycle_read: None,
            manager_state: Arc::new(std::sync::atomic::AtomicU8::new(MANAGER_UNKNOWN)),
            #[cfg(test)]
            cleanup_negative_observed: None,
        })
    }

    fn current_dir(&self) -> Result<Option<File>, ()> {
        let parent = File::open(self.cgroup.parent().ok_or(())?).map_err(|_| ())?;
        let name = self.cgroup.file_name().ok_or(())?;
        match openat(
            &parent,
            name,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        ) {
            Ok(dir) => Ok(Some(File::from(dir))),
            Err(Errno::NOENT) => Ok(None),
            Err(_) => Err(()),
        }
    }

    fn exists(&self) -> bool {
        !matches!(self.current_dir(), Ok(None))
    }

    fn creation_observed(&self) -> bool {
        self.bound.lock().is_ok_and(|bound| bound.is_some())
    }

    fn manager_terminal(&self) -> bool {
        if let Some(read) = self.lifecycle_read.as_ref() {
            let mut marker = [0_u8; 1];
            loop {
                match rustix::io::read(read, &mut marker) {
                    Ok(1) => {
                        let current = self.manager_state.load(Ordering::Acquire);
                        let next = match (current, marker[0]) {
                            (MANAGER_UNKNOWN, b'P') => MANAGER_REQUEST_SENT,
                            (MANAGER_UNKNOWN | MANAGER_REQUEST_SENT, b'N') => MANAGER_REJECTED,
                            (MANAGER_REQUEST_SENT, b'A') => MANAGER_ADMITTED,
                            _ => MANAGER_INVALID,
                        };
                        self.manager_state.store(next, Ordering::Release);
                    }
                    Ok(0) => {
                        // The trusted launcher exited without starting a manager
                        // request. EOF after P is ambiguous and stays pending.
                        let _ = self.manager_state.compare_exchange(
                            MANAGER_UNKNOWN,
                            MANAGER_REJECTED,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        );
                        break;
                    }
                    Err(Errno::AGAIN) => break,
                    Err(_) => {
                        self.manager_state.store(MANAGER_INVALID, Ordering::Release);
                        break;
                    }
                    Ok(_) => unreachable!(),
                }
            }
        }
        matches!(
            self.manager_state.load(Ordering::Acquire),
            MANAGER_ADMITTED | MANAGER_REJECTED
        )
    }

    fn creation_settled(&self) -> bool {
        if self.lifecycle_read.is_some() {
            self.manager_terminal()
        } else {
            self.creation_observed()
        }
    }

    fn bind(&self) -> Result<(), ()> {
        self.bind_if_present()?.then_some(()).ok_or(())
    }

    fn bind_if_present(&self) -> Result<bool, ()> {
        let Some(current) = self.current_dir()? else {
            return Ok(false);
        };
        let mut bound = self.bound.lock().map_err(|_| ())?;
        if let Some(original) = bound.as_ref() {
            let a = original.metadata().map_err(|_| ())?;
            let b = current.metadata().map_err(|_| ())?;
            if (a.dev(), a.ino()) != (b.dev(), b.ino()) {
                return Err(());
            }
        } else {
            *bound = Some(current);
        }
        Ok(true)
    }

    fn kill_bound(&self) -> Result<(), ()> {
        // A scope can first appear during cooperative cleanup or a retry.
        self.bind_if_present()?;
        let Some(current) = self.current_dir()? else {
            return Ok(());
        };
        let bound = self.bound.lock().map_err(|_| ())?;
        let original = bound.as_ref().ok_or(())?;
        let a = original.metadata().map_err(|_| ())?;
        let b = current.metadata().map_err(|_| ())?;
        if (a.dev(), a.ino()) != (b.dev(), b.ino()) {
            return Err(());
        }
        let control = openat(
            original,
            "cgroup.kill",
            OFlags::WRONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(|_| ())?;
        if rustix::io::write(&control, b"1").map_err(|_| ())? != 1 {
            return Err(());
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct AecBackendSessionOwner {
    state: Arc<Mutex<OwnerState>>,
    control: Arc<Mutex<OwnerControl>>,
    verified_frames: Arc<AtomicU64>,
    cleanup_signal_gate: Option<Arc<AtomicBool>>,
    #[cfg(test)]
    collector_panic_gate: Option<Arc<AtomicBool>>,
    #[cfg(test)]
    startup_gate: Option<Arc<AtomicBool>>,
    #[cfg(test)]
    handoff_panic: bool,
}

impl Default for AecBackendSessionOwner {
    fn default() -> Self {
        Self::new()
    }
}

impl AecBackendSessionOwner {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(OwnerState {
                status: AecBackendSessionStatus::Idle,
            })),
            control: Arc::new(Mutex::new(OwnerControl {
                task: None,
                cancel: None,
                terminal: None,
                closing: false,
            })),
            verified_frames: Arc::new(AtomicU64::new(0)),
            cleanup_signal_gate: None,
            #[cfg(test)]
            collector_panic_gate: None,
            #[cfg(test)]
            startup_gate: None,
            #[cfg(test)]
            handoff_panic: false,
        }
    }

    #[cfg(test)]
    fn with_startup_gate(gate: Arc<AtomicBool>) -> Self {
        let mut owner = Self::new();
        owner.startup_gate = Some(gate);
        owner
    }

    #[cfg(test)]
    fn with_collector_panic_gate(gate: Arc<AtomicBool>) -> Self {
        let mut owner = Self::new();
        owner.collector_panic_gate = Some(gate);
        owner
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub fn with_cleanup_signal_gate(gate: Arc<AtomicBool>) -> Self {
        let mut owner = Self::new();
        owner.cleanup_signal_gate = Some(gate);
        owner
    }

    pub async fn status(&self) -> AecBackendSessionStatus {
        self.state.lock().await.status
    }

    /// Transport progress only. It can never admit a calibration proof.
    #[allow(dead_code)]
    pub fn verified_frames(&self) -> u64 {
        self.verified_frames.load(Ordering::Acquire)
    }

    /// Closes the owner to new attempts and joins its one cleanup task.
    /// A pending return retains the task for a later shutdown call.
    #[allow(dead_code)]
    pub async fn shutdown(&self) -> AecBackendSessionStatus {
        let deadline = Instant::now() + TOTAL_CLEANUP;
        let Ok(mut control) = tokio::time::timeout_at(deadline, self.control.lock()).await else {
            return AecBackendSessionStatus::CleanupPending;
        };
        control.closing = true;
        if let Some(cancel) = &control.cancel {
            cancel.send_replace(true);
        }
        drop(control);

        loop {
            let Ok(mut control) = tokio::time::timeout_at(deadline, self.control.lock()).await
            else {
                return AecBackendSessionStatus::CleanupPending;
            };
            let task = if control
                .task
                .as_ref()
                .is_some_and(thread::JoinHandle::is_finished)
            {
                control.task.take()
            } else {
                None
            };
            let still_running = control.task.is_some();
            let terminal = control
                .terminal
                .as_ref()
                .and_then(|receiver| *receiver.borrow())
                .unwrap_or(AecBackendSessionStatus::Idle);
            drop(control);
            if let Some(task) = task {
                if task.join().is_err() {
                    if let Ok(mut state) =
                        tokio::time::timeout_at(deadline, self.state.lock()).await
                    {
                        state.status = AecBackendSessionStatus::CleanupPending;
                    }
                    return AecBackendSessionStatus::CleanupPending;
                }
                continue;
            }
            if !still_running {
                let Ok(state) = tokio::time::timeout_at(deadline, self.state.lock()).await else {
                    return AecBackendSessionStatus::CleanupPending;
                };
                if state.status == AecBackendSessionStatus::Idle {
                    return if terminal == AecBackendSessionStatus::CleanupPending {
                        AecBackendSessionStatus::Idle
                    } else {
                        terminal
                    };
                }
            }
            let now = Instant::now();
            if now >= deadline {
                return AecBackendSessionStatus::CleanupPending;
            }
            tokio::time::sleep(POLL_INTERVAL.min(deadline - now)).await;
        }
    }

    #[cfg(test)]
    pub async fn start(
        &self,
        command: Command,
    ) -> Result<AecBackendAttempt, AecBackendSessionError> {
        self.start_inner(command, None).await
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) async fn start_with_test_scope(
        &self,
        command: Command,
        cgroup: PathBuf,
        expected_session: u64,
    ) -> Result<AecBackendAttempt, AecBackendSessionError> {
        self.start_inner(
            command,
            Some(OwnedScope {
                unit: "translator-aec-test".to_owned(),
                cgroup,
                expected_session,
                bound: Arc::new(StdMutex::new(None)),
                lifecycle_read: None,
                manager_state: Arc::new(std::sync::atomic::AtomicU8::new(MANAGER_UNKNOWN)),
                cleanup_negative_observed: None,
            }),
        )
        .await
    }

    /// Owns the runner's temporary user scope as well as its process group.
    #[allow(dead_code)]
    pub async fn start_isolated_runner(&self) -> Result<AecBackendAttempt, AecBackendSessionError> {
        // The authority FD is exposed only to this fixed, checked-in launcher.
        let mut command = isolated_runner_command();
        let mut scope = OwnedScope::new()?;
        let (read_fd, write_fd) = pipe_with(PipeFlags::CLOEXEC | PipeFlags::NONBLOCK)
            .map_err(|_| AecBackendSessionError::SpawnFailed)?;
        let read = Arc::new(File::from(read_fd));
        let write = Arc::new(File::from(write_fd));
        scope.lifecycle_read = Some(read);
        command.env("TRANSLATOR_AEC_LIFECYCLE_FD", write.as_raw_fd().to_string());
        unsafe {
            command.as_std_mut().pre_exec(move || {
                fcntl_setfd(write.as_fd(), FdFlags::empty())
                    .map_err(|error| std::io::Error::from_raw_os_error(error.raw_os_error()))
            });
        }
        if scope.exists() {
            return Err(AecBackendSessionError::Busy);
        }
        command.env("TRANSLATOR_AEC_SCOPE_UNIT", &scope.unit).env(
            "TRANSLATOR_AEC_EXPECTED_SESSION",
            format!("{:016x}", scope.expected_session),
        );
        #[cfg(test)]
        {
            let stage_fd = std::env::var("TRANSLATOR_AEC_STAGE_LIFECYCLE_FD")
                .ok()
                .and_then(|value| value.parse::<i32>().ok())
                .filter(|fd| *fd >= 3)
                .ok_or(AecBackendSessionError::SpawnFailed)?;
            command.env("TRANSLATOR_AEC_STAGE_LIFECYCLE_FD", stage_fd.to_string());
            unsafe {
                command.as_std_mut().pre_exec(move || {
                    let borrowed = std::os::fd::BorrowedFd::borrow_raw(stage_fd);
                    fcntl_setfd(borrowed, FdFlags::empty())
                        .map_err(|error| std::io::Error::from_raw_os_error(error.raw_os_error()))
                });
            }
        }
        self.start_inner(command, Some(scope)).await
    }

    pub(crate) async fn start_retained_native(
        &self,
    ) -> Result<RetainedNativeAecAttempt, AecBackendSessionError> {
        let mut scope = OwnedScope::new()?;
        let (read_fd, write_fd) = pipe_with(PipeFlags::CLOEXEC | PipeFlags::NONBLOCK)
            .map_err(|_| AecBackendSessionError::SpawnFailed)?;
        scope.lifecycle_read = Some(Arc::new(File::from(read_fd)));
        if scope.exists() {
            return Err(AecBackendSessionError::Busy);
        }
        let (guard, endpoints) = self
            .start_launch(
                BackendLaunch::Native {
                    lifecycle_write: Arc::new(File::from(write_fd)),
                },
                Some(scope),
            )
            .await?;
        let Some(endpoints) = endpoints else {
            // Dropping the existing guard cancels its one retained cleanup owner.
            return Err(AecBackendSessionError::SpawnFailed);
        };
        Ok(RetainedNativeAecAttempt {
            guard,
            handle: endpoints.handle,
            measurement: endpoints.measurement,
            capture: endpoints.capture,
        })
    }

    async fn start_inner(
        &self,
        command: Command,
        scope: Option<OwnedScope>,
    ) -> Result<AecBackendAttempt, AecBackendSessionError> {
        self.start_launch(BackendLaunch::Transport(Box::new(command)), scope)
            .await
            .map(|(attempt, _)| attempt)
    }

    async fn start_launch(
        &self,
        mut launch: BackendLaunch,
        scope: Option<OwnedScope>,
    ) -> Result<(AecBackendAttempt, Option<NativeEndpoints>), AecBackendSessionError> {
        let mut control = self.control.lock().await;
        if control.closing {
            return Err(AecBackendSessionError::Busy);
        }
        let mut state = self.state.lock().await;
        if state.status != AecBackendSessionStatus::Idle {
            return Err(AecBackendSessionError::Busy);
        }
        if control
            .task
            .as_ref()
            .is_some_and(|task| !task.is_finished())
        {
            return Err(AecBackendSessionError::Busy);
        }
        if let Some(task) = control.task.take() {
            drop(state);
            let joined = task.join();
            state = self.state.lock().await;
            if joined.is_err() {
                state.status = AecBackendSessionStatus::CleanupPending;
                return Err(AecBackendSessionError::Busy);
            }
            if state.status != AecBackendSessionStatus::Idle {
                return Err(AecBackendSessionError::Busy);
            }
        }
        if let BackendLaunch::Transport(command) = &mut launch {
            command
                .kill_on_drop(true)
                .stdin(Stdio::null())
                .stdout(Stdio::piped());
            // The fixed native factory establishes the same private-session contract.
            unsafe {
                command.as_std_mut().pre_exec(|| {
                    rustix::process::setsid()
                        .map(|_| ())
                        .map_err(|error| std::io::Error::from_raw_os_error(error.raw_os_error()))
                });
            }
        }
        self.verified_frames.store(0, Ordering::Release);
        state.status = AecBackendSessionStatus::Running;
        drop(state);
        let (cancel_tx, cancel_rx) = watch::channel(false);
        let (terminal_tx, terminal_rx) = watch::channel(None);
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let result = Arc::new(Mutex::new(None));
        let shared = AttemptShared {
            state: Arc::clone(&self.state),
            verified_frames: Arc::clone(&self.verified_frames),
            cleanup_signal_gate: self.cleanup_signal_gate.clone(),
            #[cfg(test)]
            collector_panic_gate: self.collector_panic_gate.clone(),
            terminal_tx,
            result: Arc::clone(&result),
        };
        let guardian_shared = shared.clone();
        let guardian_scope = scope.clone();
        #[cfg(test)]
        let startup_gate = self.startup_gate.clone();
        #[cfg(test)]
        let handoff_panic = self.handoff_panic;
        let task = thread::Builder::new()
            .name("translator-aec-custody".into())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime,
                    Err(_) => {
                        guardian_shared.state.blocking_lock().status =
                            AecBackendSessionStatus::Idle;
                        let _ = started_tx.send(Err(AecBackendSessionError::SpawnFailed));
                        return;
                    }
                };
                #[cfg(test)]
                if let Some(gate) = startup_gate {
                    while gate.load(Ordering::Acquire) {
                        thread::sleep(POLL_INTERVAL);
                    }
                }
                let spawned = runtime.block_on(async {
                    match launch {
                        BackendLaunch::Transport(mut command) => command
                            .spawn()
                            .map(|child| (child, None))
                            .map_err(|_| AecBackendSessionError::SpawnFailed),
                        BackendLaunch::Native { lifecycle_write } => {
                            let owned = guardian_scope
                                .as_ref()
                                .ok_or(AecBackendSessionError::SpawnFailed)?;
                            let result = spawn_retained_native_aec(NativeAecLaunchAuthority {
                                scope_unit: owned.unit.clone(),
                                session_id: owned.expected_session,
                                lifecycle_fd: lifecycle_write.as_raw_fd(),
                            })
                            .await
                            .map(|(child, source)| (child, Some(source)))
                            .map_err(|_| AecBackendSessionError::SpawnFailed);
                            drop(lifecycle_write);
                            result
                        }
                    }
                });
                let (mut child, native) = match spawned {
                    Ok(spawned) => spawned,
                    Err(_) => {
                        runtime.block_on(async {
                            guardian_shared.state.lock().await.status =
                                AecBackendSessionStatus::Idle;
                        });
                        let _ = started_tx.send(Err(AecBackendSessionError::SpawnFailed));
                        return;
                    }
                };
                let Some(group) = child.id() else {
                    drop(native);
                    runtime.block_on(async {
                        guardian_shared.state.lock().await.status =
                            AecBackendSessionStatus::CleanupPending;
                    });
                    let _ = started_tx.send(Err(AecBackendSessionError::SpawnFailed));
                    runtime.block_on(async {
                        loop {
                            let _ = child.start_kill();
                            if child.wait().await.is_ok() {
                                guardian_shared.state.lock().await.status =
                                    AecBackendSessionStatus::Idle;
                                break;
                            }
                            tokio::time::sleep(POLL_INTERVAL).await;
                        }
                    });
                    return;
                };
                let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    #[cfg(test)]
                    if handoff_panic {
                        panic!("injected pre-handoff failure");
                    }
                    let native_start = if native.is_some() {
                        Some(started_tx)
                    } else {
                        let _ = started_tx.send(Ok(StartedBackend {
                            group,
                            native: None,
                        }));
                        None
                    };
                    runtime.block_on(run_attempt(
                        &mut child,
                        group,
                        guardian_scope.clone(),
                        cancel_rx,
                        guardian_shared.clone(),
                        native,
                        native_start,
                    ));
                }));
                if outcome.is_err() {
                    runtime.block_on(async {
                        guardian_shared.state.lock().await.status =
                            AecBackendSessionStatus::CleanupPending;
                        guardian_shared
                            .terminal_tx
                            .send_replace(Some(AecBackendSessionStatus::CleanupPending));
                        retain_cleanup(
                            &mut child,
                            group,
                            guardian_scope.as_ref(),
                            &guardian_shared,
                        )
                        .await;
                    });
                }
            })
            .map_err(|_| AecBackendSessionError::SpawnFailed);
        let task = match task {
            Ok(task) => task,
            Err(error) => {
                self.state.lock().await.status = AecBackendSessionStatus::Idle;
                return Err(error);
            }
        };
        control.cancel = Some(cancel_tx.clone());
        control.terminal = Some(terminal_rx.clone());
        control.task = Some(task);
        drop(control);
        let mut pending = PendingStart(Some(cancel_tx.clone()));
        let started = match started_rx.await {
            Ok(Ok(started)) => started,
            Ok(Err(error)) => return Err(error),
            Err(_) => {
                // The guardian may already have verified cleanup. A late API
                // observer must not replace its authoritative lifecycle state.
                return Err(AecBackendSessionError::SpawnFailed);
            }
        };
        let attempt = AecBackendAttempt {
            cancel_tx,
            terminal_rx,
            result,
            scope,
            group: started.group,
        };
        pending.0 = None;
        Ok((attempt, started.native))
    }
}

fn isolated_runner_command() -> Command {
    let mut command = Command::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../scripts/translator-aec-backend-check"
    ));
    command
        .arg("--isolated")
        .arg("--stream")
        .arg("--trial-frames")
        .arg("4500")
        .stderr(Stdio::null());
    command
}

pub struct AecBackendAttempt {
    cancel_tx: watch::Sender<bool>,
    terminal_rx: watch::Receiver<Option<AecBackendSessionStatus>>,
    #[allow(dead_code)]
    result: Arc<Mutex<Option<AecBackendTransportResult>>>,
    #[allow(dead_code)] // Diagnostic scope readback; native custody stays guardian-owned.
    scope: Option<OwnedScope>,
    #[allow(dead_code)]
    group: u32,
}

impl AecBackendAttempt {
    #[allow(dead_code)]
    pub fn owned_scope(&self) -> Option<(&str, &std::path::Path)> {
        self.scope
            .as_ref()
            .map(|scope| (scope.unit.as_str(), scope.cgroup.as_path()))
    }

    #[allow(dead_code)]
    pub fn owned_group(&self) -> u32 {
        self.group
    }

    pub async fn cancel(&self) -> AecBackendSessionStatus {
        self.cancel_tx.send_replace(true);
        self.terminal().await
    }

    #[allow(dead_code)]
    pub async fn wait(&self) -> AecBackendSessionStatus {
        self.terminal().await
    }

    /// Available only after clean EOF, successful child exit and verified cleanup.
    /// The samples remain non-admissible synthetic transport evidence.
    #[allow(dead_code)]
    pub async fn take_nonadmissible_result(&self) -> Option<AecBackendTransportResult> {
        if *self.terminal_rx.borrow() != Some(AecBackendSessionStatus::Completed) {
            return None;
        }
        self.result.lock().await.take()
    }

    async fn terminal(&self) -> AecBackendSessionStatus {
        let mut terminal_rx = self.terminal_rx.clone();
        loop {
            if let Some(status) = *terminal_rx.borrow_and_update() {
                return status;
            }
            if terminal_rx.changed().await.is_err() {
                return AecBackendSessionStatus::CleanupPending;
            }
        }
    }
}

impl Drop for AecBackendAttempt {
    fn drop(&mut self) {
        self.cancel_tx.send_replace(true);
    }
}

async fn run_attempt(
    child: &mut Child,
    group: u32,
    scope: Option<OwnedScope>,
    mut cancel_rx: watch::Receiver<bool>,
    shared: AttemptShared,
    mut native: Option<NativeAecSource>,
    mut native_start: Option<
        tokio::sync::oneshot::Sender<Result<StartedBackend, AecBackendSessionError>>,
    >,
) {
    let started = Instant::now();
    let mut progress = started;
    let mut wire = AecBackendWire::new();
    let mut windows = Vec::new();
    let mut stdout = child.stdout.take();
    #[cfg(test)]
    if shared
        .collector_panic_gate
        .as_ref()
        .is_some_and(|gate| gate.load(Ordering::Acquire))
    {
        panic!("injected collector failure");
    }
    let mut buffer = [0_u8; 64 * 1024];
    let outcome = if let Some(source) = native.as_mut() {
        collect_native(
            source,
            group,
            scope.as_ref(),
            &mut cancel_rx,
            &shared,
            &mut native_start,
        )
        .await
    } else {
        loop {
            if scope
                .as_ref()
                .is_some_and(|owned| owned.bind_if_present().is_err())
            {
                break AecBackendSessionStatus::Invalidated;
            }
            if *cancel_rx.borrow_and_update() {
                break AecBackendSessionStatus::Cancelled;
            }
            let Some(reader) = stdout.as_mut() else {
                break AecBackendSessionStatus::SourceFailed;
            };
            tokio::select! {
                changed = cancel_rx.changed() => {
                    if changed.is_err() || *cancel_rx.borrow_and_update() {
                        break AecBackendSessionStatus::Cancelled;
                    }
                }
                read = reader.read(&mut buffer) => {
                    match read {
                        Ok(0) => {
                            if wire.accepted_frames() == 0 {
                                break AecBackendSessionStatus::SourceFailed;
                            }
                            if wire.finish().is_err() {
                                break AecBackendSessionStatus::Invalidated;
                            }
                            if wire.accepted_frames() != TARGET_FRAMES
                                || windows.len() != TARGET_WINDOWS
                            {
                                break AecBackendSessionStatus::SourceFailed;
                            }
                            // Observe exit without reaping the leader: its PID must remain
                            // reserved until all descendants receive the final group signal.
                            let exited = loop {
                                match peek_child_exit(group) {
                                    Ok(Some(true)) => break AecBackendSessionStatus::Completed,
                                    Ok(Some(false)) | Err(()) => break AecBackendSessionStatus::SourceFailed,
                                    Ok(None) => {}
                                }
                                tokio::select! {
                                    changed = cancel_rx.changed() => {
                                        if changed.is_err() || *cancel_rx.borrow_and_update() {
                                            break AecBackendSessionStatus::Cancelled;
                                        }
                                    }
                                    _ = tokio::time::sleep(POLL_INTERVAL) => {}
                                    _ = tokio::time::sleep_until(progress + NO_PROGRESS_BUDGET) => {
                                        break AecBackendSessionStatus::NoProgress;
                                    }
                                    _ = tokio::time::sleep_until(started + RUN_BUDGET) => {
                                        break AecBackendSessionStatus::TimedOut;
                                    }
                                }
                            };
                            break exited;
                        }
                        Err(_) => break AecBackendSessionStatus::SourceFailed,
                        Ok(count) => {
                            let before = wire.accepted_frames();
                            let decoded = match wire.push_bytes(&buffer[..count]) {
                                Ok(decoded) => decoded,
                                Err(_) => break AecBackendSessionStatus::Invalidated,
                            };
                            if scope.as_ref().is_some_and(|owned| {
                                wire.identity().is_some_and(|identity| {
                                    identity.session != owned.expected_session
                                        || identity.generation != owned.expected_session
                                }) || wire.links().is_some_and(|links| !private_link_shape(links))
                            }) {
                                break AecBackendSessionStatus::Invalidated;
                            }
                            windows.extend(decoded);
                            if wire.accepted_frames() > TARGET_FRAMES
                                || windows.len() > TARGET_WINDOWS
                            {
                                break AecBackendSessionStatus::Invalidated;
                            }
                            if wire.accepted_frames() > before {
                                if scope.as_ref().is_some_and(|owned| owned.bind().is_err()) {
                                    break AecBackendSessionStatus::Invalidated;
                                }
                                shared.verified_frames.store(wire.accepted_frames(), Ordering::Release);
                                progress = Instant::now();
                            }
                        }
                    }
                }
                _ = tokio::time::sleep_until(progress + NO_PROGRESS_BUDGET) => {
                    break AecBackendSessionStatus::NoProgress;
                }
                _ = tokio::time::sleep_until(started + RUN_BUDGET) => {
                    break AecBackendSessionStatus::TimedOut;
                }
            }
        }
    };

    if let Some(started) = native_start.take() {
        let _ = started.send(Err(AecBackendSessionError::SpawnFailed));
    }

    shared.state.lock().await.status = AecBackendSessionStatus::CleanupPending;
    if let Some(scope) = scope.as_ref() {
        let _ = scope.bind_if_present();
    }
    // Dropping a reader cannot leave a blocking read holding the process owner.
    drop(stdout);
    // Native source drop closes command stdin for cooperative STOP; only this
    // guardian can signal, reap or release the actual child and scope.
    drop(native);
    if let Some(exit_status) =
        cleanup_group(child, group, scope.as_ref(), &shared.cleanup_signal_gate).await
    {
        let terminal = if outcome == AecBackendSessionStatus::Completed && !exit_status.success() {
            AecBackendSessionStatus::SourceFailed
        } else if outcome == AecBackendSessionStatus::Completed && *cancel_rx.borrow_and_update() {
            AecBackendSessionStatus::Cancelled
        } else {
            outcome
        };
        if terminal == AecBackendSessionStatus::Completed {
            *shared.result.lock().await = Some(AecBackendTransportResult { windows });
        }
        shared.state.lock().await.status = AecBackendSessionStatus::Idle;
        shared.terminal_tx.send_replace(Some(terminal));
        return;
    }

    // Return the bounded failure to callers, but keep custody and deny restart.
    shared
        .terminal_tx
        .send_replace(Some(AecBackendSessionStatus::CleanupPending));
    retain_cleanup(child, group, scope.as_ref(), &shared).await;
}

async fn collect_native(
    source: &mut NativeAecSource,
    group: u32,
    scope: Option<&OwnedScope>,
    cancel_rx: &mut watch::Receiver<bool>,
    shared: &AttemptShared,
    started: &mut Option<
        tokio::sync::oneshot::Sender<Result<StartedBackend, AecBackendSessionError>>,
    >,
) -> AecBackendSessionStatus {
    let Some(scope) = scope else {
        return AecBackendSessionStatus::Invalidated;
    };
    let mut progress = Instant::now();
    let mut ready = false;
    let mut adc_frames = 0;
    loop {
        if *cancel_rx.borrow_and_update() {
            return AecBackendSessionStatus::Cancelled;
        }
        if scope.bind_if_present().is_err()
            || (ready
                && (scope.bind().is_err()
                    || !scope.manager_terminal()
                    || scope.manager_state.load(Ordering::Acquire) != MANAGER_ADMITTED))
        {
            return AecBackendSessionStatus::Invalidated;
        }
        let deadline = progress + NO_PROGRESS_BUDGET;
        tokio::select! {
            biased;
            changed = cancel_rx.changed() => {
                if changed.is_err() || *cancel_rx.borrow_and_update() {
                    return AecBackendSessionStatus::Cancelled;
                }
            }
            _ = tokio::time::sleep_until(deadline) => {
                return AecBackendSessionStatus::NoProgress;
            }
            event = source.next_event(deadline) => {
                match event {
                    Ok(NativeAecEvent::Ready(identity)) => {
                        let session_is_private = group_pid(group).is_some_and(|pid| {
                            rustix::process::getsid(Some(pid)).is_ok_and(|sid| sid == pid)
                                && rustix::process::getpgid(Some(pid)).is_ok_and(|pgid| pgid == pid)
                        });
                        if ready
                            || !identity.graph.is_valid()
                            || !matches!(identity.graph, AecGraphIdentity::Native { session_id, generation, .. }
                                if session_id == scope.expected_session && generation != 0)
                            || !session_is_private
                            || !scope.manager_terminal()
                            || scope.manager_state.load(Ordering::Acquire) != MANAGER_ADMITTED
                            || scope.bind().is_err()
                        {
                            return AecBackendSessionStatus::Invalidated;
                        }
                        ready = true;
                    }
                    Ok(NativeAecEvent::Progress { adc_frames: observed }) => {
                        if !ready || observed <= adc_frames {
                            return AecBackendSessionStatus::Invalidated;
                        }
                        adc_frames = observed;
                        shared.verified_frames.store(observed, Ordering::Release);
                        progress = Instant::now();
                        if let Some(sender) = started.take() {
                            let endpoints = match (source.take_measurement(), source.take_capture()) {
                                (Ok(measurement), Ok(capture)) => NativeEndpoints {
                                    handle: source.handle(), measurement, capture,
                                },
                                _ => return AecBackendSessionStatus::Invalidated,
                            };
                            if sender.send(Ok(StartedBackend { group, native: Some(endpoints) })).is_err() {
                                return AecBackendSessionStatus::Cancelled;
                            }
                        }
                    }
                    Ok(NativeAecEvent::Poisoned) => return AecBackendSessionStatus::Invalidated,
                    Err(translator_audio::NativeAecError::Deadline) => return AecBackendSessionStatus::NoProgress,
                    Err(_) => return AecBackendSessionStatus::SourceFailed,
                }
            }
        }
    }
}

async fn retain_cleanup(
    child: &mut Child,
    group: u32,
    scope: Option<&OwnedScope>,
    shared: &AttemptShared,
) {
    loop {
        // The unreaped leader pins the numeric PGID until every member is gone.
        if child.id() == Some(group) && scope.is_none_or(OwnedScope::creation_settled) {
            let _ = signal_group_for_attempt(group, Signal::KILL, &shared.cleanup_signal_gate);
        }
        if let Some(scope) = scope {
            // An observed, inode-bound scope is ours even while the manager reply is pending.
            let _ = scope.kill_bound();
        }
        if wait_for_cleanup(child, group, scope, Instant::now() + TOTAL_CLEANUP)
            .await
            .is_some()
        {
            shared.state.lock().await.status = AecBackendSessionStatus::Idle;
            return;
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

fn private_link_shape(links: [AecBackendLink; 3]) -> bool {
    links[0].peer_node == links[1].peer_node
        && links[0].peer_port != links[1].peer_port
        && links[2].peer_node != links[0].peer_node
}

fn group_pid(id: u32) -> Option<Pid> {
    i32::try_from(id).ok().and_then(Pid::from_raw)
}

fn group_exists(id: u32) -> Result<bool, ()> {
    let pid = group_pid(id).ok_or(())?;
    match test_kill_process_group(pid) {
        Ok(()) => Ok(true),
        Err(Errno::SRCH) => Ok(false),
        Err(_) => Err(()),
    }
}

fn peek_child_exit(id: u32) -> Result<Option<bool>, ()> {
    let pid = group_pid(id).ok_or(())?;
    let status = waitid(
        WaitId::Pid(pid),
        WaitIdOptions::NOHANG | WaitIdOptions::NOWAIT | WaitIdOptions::EXITED,
    )
    .map_err(|_| ())?;
    Ok(status.map(|observed| observed.exit_status() == Some(0)))
}

fn signal_group(id: u32, signal: Signal) -> Result<(), ()> {
    let pid = group_pid(id).ok_or(())?;
    match kill_process_group(pid, signal) {
        Ok(()) | Err(Errno::SRCH) => Ok(()),
        Err(_) => Err(()),
    }
}

fn signal_group_for_attempt(
    group: u32,
    signal: Signal,
    gate: &Option<Arc<AtomicBool>>,
) -> Result<(), ()> {
    if gate
        .as_ref()
        .is_some_and(|held| held.load(Ordering::Acquire))
    {
        return Ok(());
    }
    signal_group(group, signal)
}

async fn cleanup_group(
    child: &mut Child,
    group: u32,
    scope: Option<&OwnedScope>,
    signal_gate: &Option<Arc<AtomicBool>>,
) -> Option<ExitStatus> {
    let start = Instant::now();
    let deadline = start + TOTAL_CLEANUP;
    if signal_group_for_attempt(group, Signal::TERM, signal_gate).is_err() {
        return None;
    }
    // Keep the group leader unreaped until the final group signal. Its PID then
    // cannot be reused for an unrelated group while we still address the PGID.
    tokio::time::sleep_until(start + COOPERATIVE_CLEANUP).await;
    if scope.is_none_or(OwnedScope::creation_settled)
        && signal_group_for_attempt(group, Signal::KILL, signal_gate).is_err()
    {
        return None;
    }
    if let Some(scope) = scope {
        let _ = scope.kill_bound();
    }
    wait_for_cleanup(child, group, scope, deadline).await
}

fn session_has_other_members(session: u32) -> Result<bool, ()> {
    for entry in std::fs::read_dir("/proc").map_err(|_| ())? {
        let entry = entry.map_err(|_| ())?;
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<u32>().ok())
        else {
            continue;
        };
        let stat = match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
            Ok(stat) => stat,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(_) => return Err(()),
        };
        let after_comm = stat.rsplit_once(") ").ok_or(())?.1;
        let mut fields = after_comm.split_whitespace();
        let _state = fields.next().ok_or(())?;
        let _parent = fields.next().ok_or(())?;
        let _process_group = fields.next().ok_or(())?;
        let process_session = fields.next().ok_or(())?.parse::<u32>().map_err(|_| ())?;
        if process_session == session && pid != session {
            return Ok(true);
        }
    }
    Ok(false)
}

async fn wait_for_cleanup(
    child: &mut Child,
    group: u32,
    scope: Option<&OwnedScope>,
    deadline: Instant,
) -> Option<ExitStatus> {
    loop {
        // Keep the session leader unreaped until every other process in its
        // private session is gone. A descendant in another PGID could
        // otherwise rejoin the leader's group after a group-only scan.
        // Absence alone is not a completed scope-creation transaction: the
        // manager may still create this unique unit after the first negative poll.
        let scope_cleared = if let Some(owned) = scope {
            match owned.bind_if_present() {
                Ok(true) => {
                    let _ = owned.kill_bound();
                    false
                }
                Ok(false) => {
                    #[cfg(test)]
                    if !owned.creation_observed()
                        && let Some(gate) = owned.cleanup_negative_observed.as_ref()
                    {
                        gate.store(true, Ordering::Release);
                    }
                    owned.creation_settled()
                }
                Err(()) => false,
            }
        } else {
            true
        };
        if scope_cleared
            && peek_child_exit(group).ok().flatten().is_some()
            && !session_has_other_members(group).ok()?
        {
            let status = child.try_wait().ok()??;
            if matches!(group_exists(group), Ok(false)) {
                return Some(status);
            }
            return None;
        }
        let now = Instant::now();
        if now >= deadline {
            return None;
        }
        tokio::time::sleep(POLL_INTERVAL.min(deadline - now)).await;
    }
}

#[cfg(test)]
mod scope_tests {
    use super::*;

    fn fake_scope(path: PathBuf) -> OwnedScope {
        OwnedScope {
            unit: "test-scope".to_string(),
            cgroup: path,
            expected_session: 1,
            bound: Arc::new(StdMutex::new(None)),
            lifecycle_read: None,
            manager_state: Arc::new(std::sync::atomic::AtomicU8::new(MANAGER_UNKNOWN)),
            cleanup_negative_observed: None,
        }
    }

    #[test]
    fn trusted_runner_command_has_no_caller_control() {
        let command = isolated_runner_command();
        let inner = command.as_std();
        assert_eq!(
            inner.get_program(),
            std::ffi::OsStr::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../scripts/translator-aec-backend-check"
            ))
        );
        assert_eq!(
            inner.get_args().collect::<Vec<_>>(),
            ["--isolated", "--stream", "--trial-frames", "4500"]
        );
        assert!(inner.get_envs().next().is_none());
    }

    #[tokio::test]
    async fn explicit_manager_rejection_releases_absent_scope_custody() {
        let root = tempfile::tempdir().unwrap();
        let mut scope = fake_scope(root.path().join("rejected.scope"));
        let (read, write) = pipe_with(PipeFlags::CLOEXEC | PipeFlags::NONBLOCK).unwrap();
        scope.lifecycle_read = Some(Arc::new(File::from(read)));
        let owner = AecBackendSessionOwner::new();
        let mut command = Command::new("/usr/bin/sleep");
        command.arg("30");
        let attempt = owner.start_inner(command, Some(scope)).await.unwrap();
        let group = attempt.owned_group();
        let cancellation = tokio::spawn(async move { attempt.cancel().await });
        tokio::time::sleep(Duration::from_millis(100)).await;
        rustix::io::write(&write, b"PN").unwrap();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(9), cancellation)
                .await
                .unwrap()
                .unwrap(),
            AecBackendSessionStatus::Cancelled
        );
        assert_eq!(owner.status().await, AecBackendSessionStatus::Idle);
        assert_eq!(group_exists(group), Ok(false));
        assert_eq!(owner.shutdown().await, AecBackendSessionStatus::Cancelled);
    }

    #[tokio::test]
    async fn admitted_then_vanished_scope_releases_custody() {
        let root = tempfile::tempdir().unwrap();
        let mut scope = fake_scope(root.path().join("vanished.scope"));
        let (read, write) = pipe_with(PipeFlags::CLOEXEC | PipeFlags::NONBLOCK).unwrap();
        scope.lifecycle_read = Some(Arc::new(File::from(read)));
        let owner = AecBackendSessionOwner::new();
        let mut command = Command::new("/usr/bin/sleep");
        command.arg("30");
        let attempt = owner.start_inner(command, Some(scope)).await.unwrap();
        let cancellation = tokio::spawn(async move { attempt.cancel().await });
        tokio::time::sleep(Duration::from_millis(100)).await;
        rustix::io::write(&write, b"PA").unwrap();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(9), cancellation)
                .await
                .unwrap()
                .unwrap(),
            AecBackendSessionStatus::Cancelled
        );
        assert_eq!(owner.status().await, AecBackendSessionStatus::Idle);
    }

    #[tokio::test]
    async fn pre_request_exit_without_marker_releases_owner_after_reap() {
        let root = tempfile::tempdir().unwrap();
        let mut scope = fake_scope(root.path().join("pre-request.scope"));
        let (read, write) = pipe_with(PipeFlags::CLOEXEC | PipeFlags::NONBLOCK).unwrap();
        scope.lifecycle_read = Some(Arc::new(File::from(read)));
        drop(write);
        let owner = AecBackendSessionOwner::new();
        let attempt = owner
            .start_inner(Command::new("/usr/bin/false"), Some(scope))
            .await
            .unwrap();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(9), attempt.wait())
                .await
                .unwrap(),
            AecBackendSessionStatus::SourceFailed
        );
        assert_eq!(owner.status().await, AecBackendSessionStatus::Idle);
    }

    #[test]
    fn sent_request_then_eof_without_terminal_remains_unknown() {
        let root = tempfile::tempdir().unwrap();
        let mut scope = fake_scope(root.path().join("unknown.scope"));
        let (read, write) = pipe_with(PipeFlags::CLOEXEC | PipeFlags::NONBLOCK).unwrap();
        scope.lifecycle_read = Some(Arc::new(File::from(read)));
        rustix::io::write(&write, b"P").unwrap();
        drop(write);
        assert!(!scope.creation_settled());
        assert!(!scope.creation_settled());
    }

    #[test]
    fn invalid_or_conflicting_manager_marker_cannot_settle_creation() {
        let root = tempfile::tempdir().unwrap();
        let mut scope = fake_scope(root.path().join("invalid.scope"));
        let (read, write) = pipe_with(PipeFlags::CLOEXEC | PipeFlags::NONBLOCK).unwrap();
        scope.lifecycle_read = Some(Arc::new(File::from(read)));
        rustix::io::write(&write, b"X").unwrap();
        assert!(!scope.creation_settled());

        let mut scope = fake_scope(root.path().join("conflict.scope"));
        let (read, write) = pipe_with(PipeFlags::CLOEXEC | PipeFlags::NONBLOCK).unwrap();
        scope.lifecycle_read = Some(Arc::new(File::from(read)));
        rustix::io::write(&write, b"PAN").unwrap();
        assert!(!scope.creation_settled());
    }

    #[tokio::test]
    async fn observed_scope_is_killed_while_manager_outcome_is_unknown() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("pending.scope");
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("cgroup.kill"), b"").unwrap();
        let mut scope = fake_scope(path.clone());
        let (read, write) = pipe_with(PipeFlags::CLOEXEC | PipeFlags::NONBLOCK).unwrap();
        scope.lifecycle_read = Some(Arc::new(File::from(read)));
        rustix::io::write(&write, b"P").unwrap();
        let mut command = Command::new("/usr/bin/sleep");
        command.arg("30").kill_on_drop(true);
        let mut child = command.spawn().unwrap();
        let group = child.id().unwrap();
        let outcome = wait_for_cleanup(
            &mut child,
            group,
            Some(&scope),
            Instant::now() + Duration::from_millis(100),
        )
        .await;
        assert!(outcome.is_none(), "unknown manager outcome ended cleanup");
        assert!(!scope.creation_settled());
        assert_eq!(std::fs::read(path.join("cgroup.kill")).unwrap(), b"1");
        child.start_kill().unwrap();
        let _ = child.wait().await.unwrap();
    }

    #[test]
    fn replaced_scope_cannot_receive_kill() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("owned.scope");
        std::fs::create_dir(&path).unwrap();
        let scope = fake_scope(path.clone());
        scope.bind().unwrap();
        std::fs::rename(&path, root.path().join("old.scope")).unwrap();
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("cgroup.kill"), b"foreign").unwrap();
        assert!(scope.kill_bound().is_err());
        assert_eq!(std::fs::read(path.join("cgroup.kill")).unwrap(), b"foreign");
    }

    #[test]
    fn unavailable_kill_control_retains_cleanup_obligation() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("owned.scope");
        std::fs::create_dir(&path).unwrap();
        let scope = fake_scope(path);
        scope.bind().unwrap();
        assert!(scope.kill_bound().is_err());
        assert!(scope.exists());
    }

    #[tokio::test]
    #[ignore = "requires a writable isolated user systemd scope"]
    async fn setsid_member_in_bound_scope_is_terminated_without_group_signal() {
        let scope = OwnedScope::new().unwrap();
        let runtime = format!("/run/user/{}", rustix::process::getuid().as_raw());
        let mut command = Command::new("/usr/bin/systemd-run");
        command
            .arg("--user")
            .arg("--scope")
            .arg("--collect")
            .arg(format!("--unit={}", scope.unit))
            .arg("/usr/bin/setsid")
            .arg("/usr/bin/sleep")
            .arg("20")
            .env("XDG_RUNTIME_DIR", &runtime)
            .env(
                "DBUS_SESSION_BUS_ADDRESS",
                format!("unix:path={runtime}/bus"),
            )
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        let mut launcher = command.spawn().unwrap();
        let entered = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if scope.exists()
                    && std::fs::read_to_string(scope.cgroup.join("cgroup.procs"))
                        .is_ok_and(|members| !members.trim().is_empty())
                {
                    break;
                }
                tokio::time::sleep(POLL_INTERVAL).await;
            }
        })
        .await;
        if entered.is_err() {
            let _ = launcher.start_kill();
            let _ = launcher.wait().await;
            panic!("setsid child did not enter its unique scope");
        }
        scope.bind().unwrap();
        scope.kill_bound().expect("bound cgroup.kill unavailable");
        tokio::time::timeout(Duration::from_secs(5), async {
            while scope.exists() {
                tokio::time::sleep(POLL_INTERVAL).await;
            }
        })
        .await
        .expect("setsid scope survived bound kill");
        let _ = tokio::time::timeout(Duration::from_secs(5), launcher.wait())
            .await
            .expect("systemd-run launcher survived scope cleanup")
            .expect("systemd-run launcher was not reaped");
    }

    #[tokio::test]
    #[ignore = "requires a writable isolated user systemd scope"]
    async fn no_frame_runner_reaps_detached_scope_member() {
        let scope = OwnedScope::new().unwrap();
        let runtime = format!("/run/user/{}", rustix::process::getuid().as_raw());
        let mut command = Command::new("/usr/bin/python3");
        command
            .arg("-I")
            .arg("-c")
            .arg(
                "import subprocess,sys,time\n".to_owned()
                    + "subprocess.Popen(['/usr/bin/systemd-run','--user','--scope','--collect',"
                    + "'--unit='+sys.argv[1],'/usr/bin/setsid','/usr/bin/sleep','20'],"
                    + "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
                    + "time.sleep(20)\n",
            )
            .arg(&scope.unit)
            .env("XDG_RUNTIME_DIR", &runtime)
            .env(
                "DBUS_SESSION_BUS_ADDRESS",
                format!("unix:path={runtime}/bus"),
            )
            .stderr(Stdio::null());
        let owner = AecBackendSessionOwner::new();
        let attempt = owner
            .start_inner(command, Some(scope.clone()))
            .await
            .unwrap();
        let entered = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if scope.exists()
                    && std::fs::read_to_string(scope.cgroup.join("cgroup.procs"))
                        .is_ok_and(|members| !members.trim().is_empty())
                {
                    break;
                }
                tokio::time::sleep(POLL_INTERVAL).await;
            }
        })
        .await;
        if entered.is_err() {
            let _ = attempt.cancel().await;
            panic!("no-frame runner did not create its unique scope");
        }
        assert_eq!(owner.verified_frames(), 0);
        let terminal = tokio::time::timeout(Duration::from_secs(9), attempt.cancel())
            .await
            .expect("no-frame scoped cleanup exceeded its budget");
        assert_eq!(terminal, AecBackendSessionStatus::Cancelled);
        assert_eq!(owner.status().await, AecBackendSessionStatus::Idle);
        assert!(!scope.exists(), "detached no-frame scope survived cleanup");
    }

    #[tokio::test]
    async fn late_scope_is_bound_before_retry_kill() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("late.scope");
        let scope = fake_scope(path.clone());
        let owner = AecBackendSessionOwner::new();
        let mut command = Command::new("/usr/bin/sleep");
        command.arg("30");
        let attempt = owner
            .start_inner(command, Some(scope.clone()))
            .await
            .unwrap();
        let late_path = path.clone();
        let creator = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(500)).await;
            std::fs::create_dir(&late_path).unwrap();
        });
        let terminal = tokio::time::timeout(Duration::from_secs(9), attempt.cancel())
            .await
            .expect("late-scope cleanup exceeded its first budget");
        creator.await.unwrap();
        assert_eq!(terminal, AecBackendSessionStatus::CleanupPending);
        assert!(scope.bound.lock().unwrap().is_some());
        assert_eq!(
            owner.status().await,
            AecBackendSessionStatus::CleanupPending
        );
        std::fs::remove_dir(&path).unwrap();
        assert_eq!(owner.shutdown().await, AecBackendSessionStatus::Idle);
    }

    #[tokio::test]
    async fn absent_scope_does_not_clear_late_creation_obligation() {
        use std::io::Write;

        struct CreatorGuard(std::process::Child);
        impl Drop for CreatorGuard {
            fn drop(&mut self) {
                if let Some(mut stdin) = self.0.stdin.take() {
                    let _ = stdin.write_all(b"x");
                }
                for _ in 0..100 {
                    if self.0.try_wait().ok().flatten().is_some() {
                        return;
                    }
                    thread::sleep(POLL_INTERVAL);
                }
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }

        struct ProcessGuard(std::os::fd::OwnedFd);
        impl Drop for ProcessGuard {
            fn drop(&mut self) {
                let _ = rustix::process::pidfd_send_signal(&self.0, Signal::KILL);
            }
        }

        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("delayed.scope");
        let mut scope = fake_scope(path.clone());
        let negative = Arc::new(AtomicBool::new(false));
        scope.cleanup_negative_observed = Some(Arc::clone(&negative));
        let mut creator = CreatorGuard(
            std::process::Command::new("/usr/bin/python3")
                .arg("-I")
                .arg("-c")
                .arg(
                    "import pathlib,sys,time\n".to_owned()
                        + "sys.stdin.buffer.read(1)\n"
                        + "p=pathlib.Path(sys.argv[1]); p.mkdir()\n"
                        + "(p/'cgroup.kill').write_bytes(b'0')\n"
                        + "for _ in range(500):\n"
                        + " if (p/'cgroup.kill').read_bytes()==b'1': break\n"
                        + " time.sleep(.02)\n"
                        + "else: sys.exit(3)\n"
                        + "(p/'cgroup.kill').unlink(); p.rmdir()\n",
                )
                .arg(&path)
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap(),
        );
        let owner = AecBackendSessionOwner::new();
        let mut command = Command::new("/usr/bin/sleep");
        command.arg("30");
        let attempt = owner
            .start_inner(command, Some(scope.clone()))
            .await
            .unwrap();
        let group = attempt.owned_group();
        let _process_guard = ProcessGuard(
            rustix::process::pidfd_open(
                group_pid(group).unwrap(),
                rustix::process::PidfdFlags::empty(),
            )
            .unwrap(),
        );
        let terminal = tokio::time::timeout(Duration::from_secs(9), attempt.cancel())
            .await
            .expect("cancellation exceeded cleanup budget");
        assert!(
            negative.load(Ordering::Acquire),
            "no cleanup-negative observation"
        );
        assert_eq!(terminal, AecBackendSessionStatus::CleanupPending);
        assert_eq!(
            owner.status().await,
            AecBackendSessionStatus::CleanupPending
        );
        assert!(attempt.take_nonadmissible_result().await.is_none());
        assert!(matches!(
            owner.start(Command::new("/usr/bin/true")).await,
            Err(AecBackendSessionError::Busy)
        ));

        creator.0.stdin.take().unwrap().write_all(b"x").unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while creator.0.try_wait().unwrap().is_none() {
                tokio::time::sleep(POLL_INTERVAL).await;
            }
        })
        .await
        .expect("guardian did not terminate late scope");
        assert!(creator.0.try_wait().unwrap().unwrap().success());
        assert!(!path.exists(), "scope remained after guardian cleanup");
        assert_eq!(owner.shutdown().await, AecBackendSessionStatus::Idle);
        assert_eq!(group_exists(group), Ok(false));
    }

    #[tokio::test]
    async fn shutdown_during_held_startup_still_cancels_within_entry_budget() {
        struct ReleaseGate(Arc<AtomicBool>);
        impl Drop for ReleaseGate {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }

        let gate = Arc::new(AtomicBool::new(true));
        let _release_on_failure = ReleaseGate(Arc::clone(&gate));
        let owner = AecBackendSessionOwner::with_startup_gate(Arc::clone(&gate));
        let start_owner = owner.clone();
        let start = tokio::spawn(async move {
            let mut command = Command::new("/usr/bin/sleep");
            command.arg("30");
            start_owner.start(command).await
        });
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if owner.control.lock().await.task.is_some() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("startup did not reach guardian");
        assert_eq!(
            tokio::time::timeout(TOTAL_CLEANUP + Duration::from_secs(1), owner.shutdown())
                .await
                .expect("shutdown exceeded its entry budget"),
            AecBackendSessionStatus::CleanupPending
        );
        assert!(!start.is_finished(), "held startup unexpectedly completed");
        gate.store(false, Ordering::Release);
        let attempt = tokio::time::timeout(Duration::from_secs(2), start)
            .await
            .expect("startup did not resume")
            .unwrap()
            .unwrap();
        let group = attempt.owned_group();
        assert_eq!(attempt.wait().await, AecBackendSessionStatus::Cancelled);
        assert_eq!(owner.shutdown().await, AecBackendSessionStatus::Cancelled);
        assert_eq!(group_exists(group), Ok(false));
    }

    #[tokio::test]
    async fn abandoned_startup_requests_cancel_without_losing_custody() {
        struct ReleaseGate(Arc<AtomicBool>);
        impl Drop for ReleaseGate {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }

        let gate = Arc::new(AtomicBool::new(true));
        let _release_on_failure = ReleaseGate(Arc::clone(&gate));
        let owner = AecBackendSessionOwner::with_startup_gate(Arc::clone(&gate));
        let start_owner = owner.clone();
        let start = tokio::spawn(async move {
            let mut command = Command::new("/usr/bin/sleep");
            command.arg("30");
            start_owner.start(command).await
        });
        tokio::time::timeout(Duration::from_secs(2), async {
            while owner.control.lock().await.task.is_none() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("startup did not reach guardian");
        start.abort();
        assert!(
            start
                .await
                .err()
                .expect("startup was not aborted")
                .is_cancelled()
        );
        gate.store(false, Ordering::Release);
        let reclaimed = tokio::time::timeout(Duration::from_secs(3), async {
            while owner.status().await != AecBackendSessionStatus::Idle {
                tokio::time::sleep(POLL_INTERVAL).await;
            }
        })
        .await;
        let terminal = owner.shutdown().await;
        assert!(
            reclaimed.is_ok(),
            "abandoned startup left its source running"
        );
        assert_eq!(terminal, AecBackendSessionStatus::Cancelled);
    }

    #[tokio::test]
    async fn late_failed_handoff_cannot_overwrite_guardian_cleanup() {
        struct ReleaseGate(Arc<AtomicBool>);
        impl Drop for ReleaseGate {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }

        let gate = Arc::new(AtomicBool::new(true));
        let _release_on_failure = ReleaseGate(Arc::clone(&gate));
        let mut owner = AecBackendSessionOwner::with_startup_gate(Arc::clone(&gate));
        owner.handoff_panic = true;
        let mut command = Command::new("/usr/bin/sleep");
        command.arg("30");
        let mut start = Box::pin(owner.start(command));
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        assert!(start.as_mut().poll(&mut context).is_pending());
        gate.store(false, Ordering::Release);
        let reclaimed = tokio::time::timeout(Duration::from_secs(3), async {
            while owner.status().await != AecBackendSessionStatus::Idle {
                tokio::time::sleep(POLL_INTERVAL).await;
            }
        })
        .await;
        let failed = matches!(start.await, Err(AecBackendSessionError::SpawnFailed));
        let after_handoff = owner.status().await;
        let terminal = owner.shutdown().await;
        assert!(
            reclaimed.is_ok(),
            "guardian did not clean up pre-handoff panic"
        );
        assert!(failed, "panicked startup unexpectedly handed off a source");
        assert_eq!(after_handoff, AecBackendSessionStatus::Idle);
        assert_eq!(terminal, AecBackendSessionStatus::Idle);
    }

    #[tokio::test]
    async fn collector_panic_keeps_process_custody() {
        let gate = Arc::new(AtomicBool::new(true));
        let owner = AecBackendSessionOwner::with_collector_panic_gate(gate);
        let mut command = Command::new("/usr/bin/sleep");
        command.arg("30");
        let attempt = owner.start(command).await.unwrap();
        let group = attempt.owned_group();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(2), attempt.wait())
                .await
                .expect("panic was not reported"),
            AecBackendSessionStatus::CleanupPending
        );
        assert_eq!(
            owner.status().await,
            AecBackendSessionStatus::CleanupPending
        );
        assert!(attempt.take_nonadmissible_result().await.is_none());
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(9), owner.shutdown())
                .await
                .expect("guardian did not finish process cleanup"),
            AecBackendSessionStatus::Idle
        );
        assert_eq!(group_exists(group), Ok(false));
    }

    #[tokio::test]
    async fn caller_runtime_drop_does_not_drop_process_custody() {
        let (sender, receiver) = std::sync::mpsc::channel();
        let caller = thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let (owner, attempt) = runtime.block_on(async {
                let owner = AecBackendSessionOwner::new();
                let mut command = Command::new("/usr/bin/sleep");
                command.arg("30");
                let attempt = owner.start(command).await.unwrap();
                (owner, attempt)
            });
            sender.send((owner, attempt)).unwrap();
            drop(runtime);
        });
        let (owner, attempt) = receiver.recv().unwrap();
        caller.join().unwrap();
        let group = attempt.owned_group();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(9), attempt.cancel())
                .await
                .expect("custody stopped with caller runtime"),
            AecBackendSessionStatus::Cancelled
        );
        assert_eq!(owner.shutdown().await, AecBackendSessionStatus::Cancelled);
        assert_eq!(group_exists(group), Ok(false));
    }

    #[tokio::test]
    async fn panicked_cleanup_task_never_reports_idle() {
        let owner = AecBackendSessionOwner::new();
        owner.state.lock().await.status = AecBackendSessionStatus::Running;
        owner.control.lock().await.task = Some(thread::spawn(|| {
            panic!("injected cleanup task failure");
        }));
        assert_eq!(
            owner.shutdown().await,
            AecBackendSessionStatus::CleanupPending
        );
        assert_eq!(
            owner.status().await,
            AecBackendSessionStatus::CleanupPending
        );
    }

    #[tokio::test(start_paused = true)]
    async fn concurrent_shutdown_calls_share_entry_deadline() {
        let owner = AecBackendSessionOwner::new();
        owner.state.lock().await.status = AecBackendSessionStatus::Running;
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let (finished_tx, finished_rx) = std::sync::mpsc::channel::<()>();
        let state = Arc::clone(&owner.state);
        owner.control.lock().await.task = Some(thread::spawn(move || {
            let _ = release_rx.recv();
            state.blocking_lock().status = AecBackendSessionStatus::Idle;
            let _ = finished_tx.send(());
        }));
        let (first_entered_tx, first_entered_rx) = tokio::sync::oneshot::channel();
        let first_owner = owner.clone();
        let first = tokio::spawn(async move {
            let _ = first_entered_tx.send(());
            first_owner.shutdown().await
        });
        first_entered_rx.await.unwrap();
        for _ in 0..100 {
            match owner.control.try_lock() {
                Ok(control) if control.closing => break,
                Err(_) => break,
                Ok(_) => tokio::task::yield_now().await,
            }
        }
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let second_owner = owner.clone();
        let second = tokio::spawn(async move {
            let _ = entered_tx.send(());
            second_owner.shutdown().await
        });
        entered_rx.await.unwrap();
        tokio::time::advance(TOTAL_CLEANUP).await;
        tokio::task::yield_now().await;
        assert!(
            first.is_finished(),
            "first shutdown exceeded its entry deadline"
        );
        assert!(
            second.is_finished(),
            "second shutdown waited another full budget"
        );
        assert_eq!(
            first.await.unwrap(),
            AecBackendSessionStatus::CleanupPending
        );
        assert_eq!(
            second.await.unwrap(),
            AecBackendSessionStatus::CleanupPending
        );
        assert!(owner.control.lock().await.task.is_some());
        assert!(matches!(
            owner.start(Command::new("/usr/bin/true")).await,
            Err(AecBackendSessionError::Busy)
        ));
        release_tx.send(()).unwrap();
        tokio::task::spawn_blocking(move || finished_rx.recv_timeout(Duration::from_secs(5)))
            .await
            .unwrap()
            .expect("owned cleanup thread did not finish");
        assert_eq!(owner.shutdown().await, AecBackendSessionStatus::Idle);
    }

    #[tokio::test]
    async fn failed_cgroup_write_keeps_owner_pending_until_scope_is_gone() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("owned.scope");
        std::fs::create_dir(&path).unwrap();
        let scope = fake_scope(path.clone());
        scope.bind().unwrap();
        let owner = AecBackendSessionOwner::new();
        let mut command = Command::new("/usr/bin/sleep");
        command.arg("30");
        let attempt = owner.start_inner(command, Some(scope)).await.unwrap();
        let status = tokio::time::timeout(Duration::from_secs(9), attempt.cancel())
            .await
            .expect("first cleanup did not return");
        assert_eq!(status, AecBackendSessionStatus::CleanupPending);
        assert_eq!(
            owner.status().await,
            AecBackendSessionStatus::CleanupPending
        );
        assert!(attempt.take_nonadmissible_result().await.is_none());
        std::fs::remove_dir(&path).unwrap();
        let drained = tokio::time::timeout(Duration::from_secs(10), owner.shutdown())
            .await
            .expect("owner failed to join recovered cleanup");
        assert_eq!(drained, AecBackendSessionStatus::Idle);
        assert_eq!(owner.status().await, AecBackendSessionStatus::Idle);
    }
}
