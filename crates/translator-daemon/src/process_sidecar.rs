use std::{
    collections::HashSet,
    env,
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
    process::Stdio,
    thread,
    time::Duration,
};

use rustix::{
    io::Errno,
    process::{Pid, Signal, kill_process_group, test_kill_process_group},
};
use tokio::process::{Child, Command};
use translator_ipc::{authenticated_request, connect_provider, provider::ProviderProbeRequest};
use uuid::Uuid;

use crate::{
    ChildState, SidecarLaunch, SidecarRuntime, SupervisorError, remove_stale_sidecar_socket,
};

const PROBE_REQUEST_SCHEMA: &str = "translator.provider.probe_request.v1";
const PROBE_RESPONSE_SCHEMA: &str = "translator.provider.probe_response.v1";
const PROBE_RETRY_INTERVAL: Duration = Duration::from_millis(10);
const SHUTDOWN_POLL_INTERVAL: Duration = Duration::from_millis(10);
const START_RETRY_BASE: Duration = Duration::from_millis(50);
pub const GRACEFUL_SHUTDOWN_TIMEOUT: Duration = Duration::from_millis(500);
const CUDA_COMPAT_LIBRARY_DIRS: [&str; 2] = [
    "/usr/local/lib/ollama/cuda_v12",
    "/home/anton/Source/uncle-freud-bot/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib",
];

pub struct ProcessSidecarRuntime {
    python: PathBuf,
    sidecar_root: PathBuf,
    socket_path: PathBuf,
    expected_uid: u32,
    child: Option<Child>,
    process_group_id: Option<u32>,
    last_reaped_pid: Option<u32>,
}

impl ProcessSidecarRuntime {
    pub fn new(
        python: PathBuf,
        sidecar_root: PathBuf,
        socket_path: PathBuf,
        expected_uid: u32,
    ) -> Result<Self, SupervisorError> {
        if !python.is_file()
            || !sidecar_root.is_dir()
            || !socket_path.is_absolute()
            || socket_path.file_name().is_none()
        {
            return Err(SupervisorError::StartFailed);
        }
        Ok(Self {
            python,
            sidecar_root,
            socket_path,
            expected_uid,
            child: None,
            process_group_id: None,
            last_reaped_pid: None,
        })
    }

    pub fn child_pid(&self) -> Option<u32> {
        self.child.as_ref().and_then(Child::id)
    }

    pub fn last_reaped_pid(&self) -> Option<u32> {
        self.last_reaped_pid
    }

    fn spawn(&mut self, launch: &SidecarLaunch) -> Result<(), SupervisorError> {
        let mut command = Command::new(&self.python);
        command
            .arg("-m")
            .arg("translator_sidecar")
            .current_dir(&self.sidecar_root)
            .env("TRANSLATOR_SIDECAR_SOCKET", &self.socket_path)
            .env("TRANSLATOR_SIDECAR_TOKEN", &launch.token)
            .env(
                "TRANSLATOR_SIDECAR_GENERATION",
                launch.generation_id.to_string(),
            )
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0)
            .kill_on_drop(true);
        if let Some(library_path) = sidecar_library_path() {
            command.env("LD_LIBRARY_PATH", library_path);
        }
        let child = command.spawn().map_err(|_| SupervisorError::StartFailed)?;
        self.process_group_id = child.id();
        self.child = Some(child);
        Ok(())
    }

    fn running_child(&mut self) -> Result<bool, SupervisorError> {
        let Some(child) = self.child.as_mut() else {
            self.clear_absent_process_group()?;
            return Ok(false);
        };
        match child.try_wait().map_err(|_| SupervisorError::StartFailed)? {
            None => Ok(true),
            Some(_) => {
                self.last_reaped_pid = child.id();
                self.child = None;
                self.clear_absent_process_group()?;
                Ok(false)
            }
        }
    }

    fn clear_absent_process_group(&mut self) -> Result<(), SupervisorError> {
        if self.process_group_id.is_some() && !process_group_exists(self.process_group_id)? {
            self.process_group_id = None;
        }
        Ok(())
    }

    async fn force_group_exit(&mut self) -> Result<ChildState, SupervisorError> {
        let Some(pid) = self.process_group_id else {
            self.child = None;
            return Ok(ChildState::Reaped);
        };
        if process_group_exists(Some(pid))? {
            signal_group(Some(pid), Signal::KILL)?;
        }
        if let Some(mut child) = self.child.take() {
            child
                .wait()
                .await
                .map_err(|_| SupervisorError::KillAndReapFailed)?;
        }
        if !wait_for_group_disappearance(Some(pid), GRACEFUL_SHUTDOWN_TIMEOUT).await? {
            return Err(SupervisorError::KillAndReapFailed);
        }
        self.last_reaped_pid = Some(pid);
        self.process_group_id = None;
        Ok(ChildState::Reaped)
    }
}

fn sidecar_library_path() -> Option<OsString> {
    merge_library_paths(
        env::var_os("LD_LIBRARY_PATH").as_deref(),
        CUDA_COMPAT_LIBRARY_DIRS
            .iter()
            .map(Path::new)
            .filter(|path| path.is_dir()),
    )
}

fn merge_library_paths<'a>(
    existing: Option<&OsStr>,
    prepended: impl IntoIterator<Item = &'a Path>,
) -> Option<OsString> {
    let mut seen = HashSet::new();
    let mut paths = Vec::new();
    for path in prepended {
        let owned = path.to_path_buf();
        if seen.insert(owned.clone()) {
            paths.push(owned);
        }
    }
    for path in existing.into_iter().flat_map(env::split_paths) {
        if seen.insert(path.clone()) {
            paths.push(path);
        }
    }
    (!paths.is_empty())
        .then(|| env::join_paths(paths).ok())
        .flatten()
}

impl SidecarRuntime for ProcessSidecarRuntime {
    async fn start(&mut self, launch: &SidecarLaunch) -> Result<(), SupervisorError> {
        if self.running_child()? {
            return Err(SupervisorError::StartFailed);
        }
        if self.process_group_id.is_some() {
            return Err(SupervisorError::StartFailed);
        }
        self.spawn(launch)
    }

    async fn probe(&mut self, launch: &SidecarLaunch) -> Result<Uuid, SupervisorError> {
        loop {
            if !self.running_child()? {
                return Err(SupervisorError::ReadinessFailed);
            }
            let Ok(mut client) = connect_provider(&self.socket_path).await else {
                tokio::time::sleep(PROBE_RETRY_INTERVAL).await;
                continue;
            };
            let request = authenticated_request(
                ProviderProbeRequest {
                    schema_version: PROBE_REQUEST_SCHEMA.into(),
                },
                &launch.token,
            )
            .map_err(|_| SupervisorError::ReadinessFailed)?;
            let response = client
                .probe(request)
                .await
                .map_err(|_| SupervisorError::ReadinessFailed)?
                .into_inner();
            if response.schema_version != PROBE_RESPONSE_SCHEMA {
                return Err(SupervisorError::ReadinessFailed);
            }
            return Uuid::parse_str(&response.generation_id)
                .map_err(|_| SupervisorError::ReadinessFailed);
        }
    }

    async fn kill_and_reap(&mut self) -> Result<ChildState, SupervisorError> {
        self.force_group_exit().await
    }

    async fn shutdown_and_reap(&mut self) -> Result<ChildState, SupervisorError> {
        let Some(pid) = self.process_group_id else {
            self.child = None;
            return Ok(ChildState::Reaped);
        };
        let mut child = self.child.take();
        if process_group_exists(Some(pid))? {
            signal_group(Some(pid), Signal::TERM)?;
            if !wait_for_group_exit(child.as_mut(), Some(pid), GRACEFUL_SHUTDOWN_TIMEOUT).await? {
                signal_group(Some(pid), Signal::KILL)?;
                if let Some(mut owned_child) = child.take() {
                    owned_child
                        .wait()
                        .await
                        .map_err(|_| SupervisorError::KillAndReapFailed)?;
                }
                if !wait_for_group_disappearance(Some(pid), GRACEFUL_SHUTDOWN_TIMEOUT).await? {
                    return Err(SupervisorError::KillAndReapFailed);
                }
            }
        }
        if let Some(mut child) = child {
            child
                .wait()
                .await
                .map_err(|_| SupervisorError::KillAndReapFailed)?;
        }
        self.last_reaped_pid = Some(pid);
        self.process_group_id = None;
        Ok(ChildState::Reaped)
    }

    async fn remove_stale_socket(
        &mut self,
        child_state: ChildState,
    ) -> Result<(), SupervisorError> {
        remove_stale_sidecar_socket(&self.socket_path, self.expected_uid, child_state)
            .map_err(|_| SupervisorError::CleanupFailed)
    }

    async fn wait_before_retry(&mut self, attempt: usize) -> Result<(), SupervisorError> {
        let multiplier = u32::try_from(attempt.max(1)).unwrap_or(u32::MAX);
        tokio::time::sleep(START_RETRY_BASE.saturating_mul(multiplier)).await;
        Ok(())
    }

    fn poll_child_state(&mut self) -> Result<ChildState, SupervisorError> {
        self.running_child().map(|running| {
            if running {
                ChildState::Running
            } else {
                ChildState::Reaped
            }
        })
    }
}

impl Drop for ProcessSidecarRuntime {
    fn drop(&mut self) {
        if let Some(pid) = self.process_group_id {
            let _ = signal_group(Some(pid), Signal::KILL);
        }
        if let Some(child) = self.child.as_mut() {
            for _ in 0..100 {
                if child.try_wait().ok().flatten().is_some() {
                    break;
                }
                thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

fn signal_group(pid: Option<u32>, signal: Signal) -> Result<(), SupervisorError> {
    let pid = process_group_pid(pid)?;
    match kill_process_group(pid, signal) {
        Ok(()) | Err(Errno::SRCH) => Ok(()),
        Err(_) => Err(SupervisorError::KillAndReapFailed),
    }
}

fn process_group_pid(pid: Option<u32>) -> Result<Pid, SupervisorError> {
    pid.and_then(|value| i32::try_from(value).ok())
        .and_then(Pid::from_raw)
        .ok_or(SupervisorError::KillAndReapFailed)
}

fn process_group_exists(pid: Option<u32>) -> Result<bool, SupervisorError> {
    match test_kill_process_group(process_group_pid(pid)?) {
        Ok(()) => Ok(true),
        Err(Errno::SRCH) => Ok(false),
        Err(_) => Err(SupervisorError::KillAndReapFailed),
    }
}

async fn wait_for_group_exit(
    mut child: Option<&mut Child>,
    pid: Option<u32>,
    timeout: Duration,
) -> Result<bool, SupervisorError> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if let Some(child) = child.as_deref_mut() {
            child
                .try_wait()
                .map_err(|_| SupervisorError::KillAndReapFailed)?;
        }
        if !process_group_exists(pid)? {
            return Ok(true);
        }
        if tokio::time::Instant::now() >= deadline {
            return Ok(false);
        }
        tokio::time::sleep(SHUTDOWN_POLL_INTERVAL).await;
    }
}

async fn wait_for_group_disappearance(
    pid: Option<u32>,
    timeout: Duration,
) -> Result<bool, SupervisorError> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if !process_group_exists(pid)? {
            return Ok(true);
        }
        if tokio::time::Instant::now() >= deadline {
            return Ok(false);
        }
        tokio::time::sleep(SHUTDOWN_POLL_INTERVAL).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_compat_paths_precede_and_deduplicate_existing_library_path() {
        let value = merge_library_paths(
            Some(OsStr::new("/usr/lib:/cuda")),
            [Path::new("/cuda"), Path::new("/cudnn")],
        )
        .expect("merged library path");

        assert_eq!(value, OsStr::new("/cuda:/cudnn:/usr/lib"));
    }
}
