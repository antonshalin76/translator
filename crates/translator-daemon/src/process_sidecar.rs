use std::{
    collections::{HashSet, VecDeque},
    env,
    ffi::{OsStr, OsString},
    fs::{self, Metadata},
    os::unix::{
        ffi::OsStrExt,
        fs::{MetadataExt, PermissionsExt},
    },
    path::Component,
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
const CUDA_COMPAT_LIBRARY_DIRS: [&str; 1] = ["/usr/local/lib/ollama/cuda_v12"];
const CUDA_LIBRARY_TREE_MAX_ENTRIES: usize = 4096;
const CUDA_PATH_MAX_SYMLINKS: usize = 40;
const GLIBC_TUNABLES: &str = "GLIBC_TUNABLES";

#[derive(Clone, Copy)]
struct CudaPathPolicy {
    user_uid: u32,
    root_uid: u32,
}

#[derive(Clone, Copy)]
enum ExpectedPathKind {
    Directory,
    File,
}

enum ResolutionComponent {
    Root,
    Parent,
    Normal(OsString),
}

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
        sanitize_dynamic_loader_environment(&mut command);
        if let Some(library_path) = sidecar_library_path(self.expected_uid)? {
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

fn sidecar_library_path(expected_uid: u32) -> Result<Option<OsString>, SupervisorError> {
    let configured = env::var_os("TRANSLATOR_CUDA_LIBRARY_PATH");
    cuda_library_path(
        configured.as_deref().filter(|value| !value.is_empty()),
        CUDA_COMPAT_LIBRARY_DIRS.iter().map(Path::new),
        CudaPathPolicy {
            user_uid: expected_uid,
            root_uid: 0,
        },
    )
}

fn sanitize_dynamic_loader_environment(command: &mut Command) {
    remove_dynamic_loader_environment(command, env::vars_os().map(|(name, _)| name));
}

fn remove_dynamic_loader_environment(
    command: &mut Command,
    inherited_names: impl IntoIterator<Item = OsString>,
) {
    command.env_remove(GLIBC_TUNABLES);
    for name in inherited_names {
        if name.as_bytes().starts_with(b"LD_") {
            command.env_remove(name);
        }
    }
}

fn cuda_library_path<'a>(
    configured: Option<&OsStr>,
    defaults: impl IntoIterator<Item = &'a Path>,
    policy: CudaPathPolicy,
) -> Result<Option<OsString>, SupervisorError> {
    let mut seen = HashSet::new();
    let mut admitted = Vec::new();
    for path in configured.into_iter().flat_map(env::split_paths) {
        let resolved = validate_cuda_library_directory(&path, policy)?;
        if seen.insert(resolved.clone()) {
            admitted.push(resolved);
        }
    }
    for path in defaults {
        match fs::symlink_metadata(path) {
            Ok(_) => {
                let resolved = validate_cuda_library_directory(path, policy)?;
                if seen.insert(resolved.clone()) {
                    admitted.push(resolved);
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => return Err(SupervisorError::StartFailed),
        }
    }
    if admitted.is_empty() {
        Ok(None)
    } else {
        env::join_paths(admitted)
            .map(Some)
            .map_err(|_| SupervisorError::StartFailed)
    }
}

fn validate_cuda_library_directory(
    path: &Path,
    policy: CudaPathPolicy,
) -> Result<PathBuf, SupervisorError> {
    let resolved = secure_resolve(path, ExpectedPathKind::Directory, policy)?;
    validate_cuda_library_tree(&resolved, policy)?;
    Ok(resolved)
}

fn secure_resolve(
    path: &Path,
    expected: ExpectedPathKind,
    policy: CudaPathPolicy,
) -> Result<PathBuf, SupervisorError> {
    if !path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(SupervisorError::StartFailed);
    }
    let mut pending = VecDeque::from(resolution_components(path)?);
    let mut resolved = PathBuf::from("/");
    let mut symlinks = 0usize;
    while let Some(component) = pending.pop_front() {
        match component {
            ResolutionComponent::Root => {
                resolved = PathBuf::from("/");
                let metadata =
                    fs::symlink_metadata(&resolved).map_err(|_| SupervisorError::StartFailed)?;
                validate_directory(&metadata, true, policy)?;
            }
            ResolutionComponent::Parent => {
                resolved.pop();
                if resolved.as_os_str().is_empty() {
                    resolved.push("/");
                }
            }
            ResolutionComponent::Normal(name) => {
                let candidate = resolved.join(name);
                let metadata =
                    fs::symlink_metadata(&candidate).map_err(|_| SupervisorError::StartFailed)?;
                if metadata.file_type().is_symlink() {
                    validate_symlink(&metadata, policy)?;
                    symlinks = symlinks
                        .checked_add(1)
                        .ok_or(SupervisorError::StartFailed)?;
                    if symlinks > CUDA_PATH_MAX_SYMLINKS {
                        return Err(SupervisorError::StartFailed);
                    }
                    let target =
                        fs::read_link(&candidate).map_err(|_| SupervisorError::StartFailed)?;
                    for target_component in resolution_components(&target)?.into_iter().rev() {
                        pending.push_front(target_component);
                    }
                } else {
                    if !pending.is_empty() {
                        validate_directory(&metadata, true, policy)?;
                    }
                    resolved = candidate;
                }
            }
        }
    }

    let metadata = fs::symlink_metadata(&resolved).map_err(|_| SupervisorError::StartFailed)?;
    match expected {
        ExpectedPathKind::Directory => validate_directory(&metadata, false, policy)?,
        ExpectedPathKind::File => validate_file(&metadata, policy)?,
    }
    let canonical = fs::canonicalize(path).map_err(|_| SupervisorError::StartFailed)?;
    if canonical != resolved {
        return Err(SupervisorError::StartFailed);
    }
    Ok(resolved)
}

fn resolution_components(path: &Path) -> Result<Vec<ResolutionComponent>, SupervisorError> {
    path.components()
        .filter_map(|component| match component {
            Component::RootDir => Some(Ok(ResolutionComponent::Root)),
            Component::ParentDir => Some(Ok(ResolutionComponent::Parent)),
            Component::Normal(name) => Some(Ok(ResolutionComponent::Normal(name.to_os_string()))),
            Component::CurDir => None,
            Component::Prefix(_) => Some(Err(SupervisorError::StartFailed)),
        })
        .collect()
}

fn trusted_owner(metadata: &Metadata, policy: CudaPathPolicy) -> bool {
    metadata.uid() == policy.user_uid || metadata.uid() == policy.root_uid
}

fn validate_directory(
    metadata: &Metadata,
    traversal: bool,
    policy: CudaPathPolicy,
) -> Result<(), SupervisorError> {
    let mode = metadata.permissions().mode();
    let writable = mode & 0o022 != 0;
    let root_sticky =
        traversal && metadata.uid() == policy.root_uid && mode & libc_mode::STICKY != 0;
    if !metadata.is_dir()
        || !trusted_owner(metadata, policy)
        || (writable && !root_sticky)
        || mode & libc_mode::SET_ID != 0
    {
        return Err(SupervisorError::StartFailed);
    }
    Ok(())
}

fn validate_file(metadata: &Metadata, policy: CudaPathPolicy) -> Result<(), SupervisorError> {
    let mode = metadata.permissions().mode();
    if !metadata.is_file() || !trusted_owner(metadata, policy) || mode & libc_mode::UNSAFE_FILE != 0
    {
        return Err(SupervisorError::StartFailed);
    }
    Ok(())
}

fn validate_symlink(metadata: &Metadata, policy: CudaPathPolicy) -> Result<(), SupervisorError> {
    if !metadata.file_type().is_symlink() || !trusted_owner(metadata, policy) {
        return Err(SupervisorError::StartFailed);
    }
    Ok(())
}

fn validate_cuda_library_tree(root: &Path, policy: CudaPathPolicy) -> Result<(), SupervisorError> {
    let mut pending = vec![root.to_path_buf()];
    let mut visited = HashSet::new();
    let mut entries = 0usize;
    while let Some(directory) = pending.pop() {
        let before = fs::symlink_metadata(&directory).map_err(|_| SupervisorError::StartFailed)?;
        validate_directory(&before, false, policy)?;
        if !visited.insert((before.dev(), before.ino())) {
            continue;
        }
        let children = fs::read_dir(&directory).map_err(|_| SupervisorError::StartFailed)?;
        for child in children {
            let child = child.map_err(|_| SupervisorError::StartFailed)?.path();
            entries = entries.checked_add(1).ok_or(SupervisorError::StartFailed)?;
            if entries > CUDA_LIBRARY_TREE_MAX_ENTRIES {
                return Err(SupervisorError::StartFailed);
            }
            let metadata =
                fs::symlink_metadata(&child).map_err(|_| SupervisorError::StartFailed)?;
            if metadata.file_type().is_symlink() {
                validate_symlink(&metadata, policy)?;
                let target = fs::metadata(&child).map_err(|_| SupervisorError::StartFailed)?;
                if target.is_dir() {
                    pending.push(secure_resolve(&child, ExpectedPathKind::Directory, policy)?);
                } else if target.is_file() {
                    secure_resolve(&child, ExpectedPathKind::File, policy)?;
                } else {
                    return Err(SupervisorError::StartFailed);
                }
            } else if metadata.is_dir() {
                validate_directory(&metadata, false, policy)?;
                pending.push(child);
            } else if metadata.is_file() {
                validate_file(&metadata, policy)?;
            } else {
                return Err(SupervisorError::StartFailed);
            }
        }
        let after = fs::symlink_metadata(&directory).map_err(|_| SupervisorError::StartFailed)?;
        if metadata_identity(&before) != metadata_identity(&after) {
            return Err(SupervisorError::StartFailed);
        }
    }
    Ok(())
}

fn metadata_identity(metadata: &Metadata) -> (u64, u64, u32, u32, u32, u64, i64, i64, i64, i64) {
    (
        metadata.dev(),
        metadata.ino(),
        metadata.mode(),
        metadata.uid(),
        metadata.gid(),
        metadata.nlink(),
        metadata.mtime(),
        metadata.mtime_nsec(),
        metadata.ctime(),
        metadata.ctime_nsec(),
    )
}

mod libc_mode {
    pub const STICKY: u32 = 0o1000;
    pub const SET_ID: u32 = 0o6000;
    pub const UNSAFE_FILE: u32 = 0o6022;
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
    use std::os::unix::ffi::OsStringExt;
    use tempfile::tempdir;

    fn test_policy(path: &Path) -> CudaPathPolicy {
        CudaPathPolicy {
            user_uid: fs::metadata(path).expect("test path metadata").uid(),
            root_uid: fs::metadata("/").expect("root metadata").uid(),
        }
    }

    fn make_private_directory(path: &Path) {
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .expect("private test directory");
    }

    #[test]
    fn cuda_compat_paths_are_canonical_and_deduplicated() {
        let configured = tempdir().expect("configured CUDA directory");
        let default = tempdir().expect("default CUDA directory");
        make_private_directory(configured.path());
        make_private_directory(default.path());
        let configured_value =
            env::join_paths([configured.path(), default.path()]).expect("configured CUDA paths");

        let value = cuda_library_path(
            Some(&configured_value),
            [default.path(), configured.path()],
            test_policy(configured.path()),
        )
        .expect("validated CUDA path")
        .expect("non-empty CUDA path");

        assert_eq!(
            env::split_paths(&value).collect::<Vec<_>>(),
            vec![
                configured.path().canonicalize().expect("configured path"),
                default.path().canonicalize().expect("default path"),
            ]
        );
    }

    #[test]
    fn operator_cuda_paths_must_be_absolute_and_existing() {
        let configured = tempdir().expect("configured CUDA directory");
        make_private_directory(configured.path());
        let relative = env::join_paths([Path::new("relative-cuda"), configured.path()])
            .expect("relative CUDA path list");

        assert!(
            cuda_library_path(
                Some(&relative),
                std::iter::empty(),
                test_policy(configured.path())
            )
            .is_err()
        );

        let missing = configured.path().join("missing-cuda");
        assert!(
            cuda_library_path(
                Some(missing.as_os_str()),
                std::iter::empty(),
                test_policy(configured.path())
            )
            .is_err()
        );
        assert_eq!(CUDA_COMPAT_LIBRARY_DIRS, ["/usr/local/lib/ollama/cuda_v12"]);
    }

    #[test]
    fn absent_portable_cuda_default_is_optional() {
        let configured = tempdir().expect("configured CUDA directory");
        make_private_directory(configured.path());
        let missing = configured.path().join("missing-default");

        assert_eq!(
            cuda_library_path(None, [missing.as_path()], test_policy(configured.path()))
                .expect("optional CUDA default"),
            None
        );
    }

    #[test]
    fn unsafe_operator_cuda_directory_is_rejected() {
        let configured = tempdir().expect("configured CUDA directory");
        fs::set_permissions(configured.path(), fs::Permissions::from_mode(0o777))
            .expect("world-writable CUDA directory");

        assert!(
            validate_cuda_library_directory(configured.path(), test_policy(configured.path()))
                .is_err()
        );
    }

    #[test]
    fn unsafe_shadow_library_in_cuda_directory_is_rejected() {
        let configured = tempdir().expect("configured CUDA directory");
        make_private_directory(configured.path());
        let shadow = configured.path().join("libpython-shadow.so");
        fs::write(&shadow, b"not a trusted library").expect("shadow library");
        fs::set_permissions(&shadow, fs::Permissions::from_mode(0o666))
            .expect("world-writable shadow library");

        assert!(
            validate_cuda_library_directory(configured.path(), test_policy(configured.path()))
                .is_err()
        );
    }

    #[test]
    fn unsafe_cuda_symlink_target_is_rejected() {
        let root = tempdir().expect("CUDA test root");
        make_private_directory(root.path());
        let cuda = root.path().join("cuda");
        let unsafe_parent = root.path().join("unsafe-parent");
        fs::create_dir(&cuda).expect("CUDA directory");
        fs::create_dir(&unsafe_parent).expect("unsafe target parent");
        make_private_directory(&cuda);
        fs::set_permissions(&unsafe_parent, fs::Permissions::from_mode(0o777))
            .expect("world-writable target parent");
        let target = unsafe_parent.join("libcudart.so.12.0");
        fs::write(&target, b"synthetic library").expect("target library");
        fs::set_permissions(&target, fs::Permissions::from_mode(0o600))
            .expect("private target library");
        std::os::unix::fs::symlink(&target, cuda.join("libcudart.so.12"))
            .expect("CUDA soname symlink");

        assert!(validate_cuda_library_directory(&cuda, test_policy(&cuda)).is_err());
    }

    #[test]
    fn unsafe_intermediate_cuda_symlink_parent_is_rejected() {
        let root = tempdir().expect("CUDA test root");
        make_private_directory(root.path());
        let cuda = root.path().join("cuda");
        let unsafe_parent = root.path().join("replaceable-link-parent");
        let safe_target_parent = root.path().join("safe-target-parent");
        fs::create_dir(&cuda).expect("CUDA directory");
        fs::create_dir(&unsafe_parent).expect("unsafe link parent");
        fs::create_dir(&safe_target_parent).expect("safe target parent");
        make_private_directory(&cuda);
        make_private_directory(&safe_target_parent);
        fs::set_permissions(&unsafe_parent, fs::Permissions::from_mode(0o777))
            .expect("world-writable link parent");
        let target = safe_target_parent.join("libcudart.so.12.0");
        fs::write(&target, b"synthetic library").expect("target library");
        fs::set_permissions(&target, fs::Permissions::from_mode(0o600))
            .expect("private target library");
        let intermediate = unsafe_parent.join("replaceable-link");
        std::os::unix::fs::symlink(&target, &intermediate)
            .expect("replaceable intermediate symlink");
        std::os::unix::fs::symlink(&intermediate, cuda.join("libcudart.so.12"))
            .expect("CUDA soname symlink");

        assert!(validate_cuda_library_directory(&cuda, test_policy(&cuda)).is_err());
    }

    #[test]
    fn trusted_cuda_soname_symlink_is_accepted() {
        let root = tempdir().expect("CUDA test root");
        make_private_directory(root.path());
        let cuda = root.path().join("cuda");
        fs::create_dir(&cuda).expect("CUDA directory");
        make_private_directory(&cuda);
        let target = cuda.join("libcudart.so.12.0");
        fs::write(&target, b"synthetic library").expect("target library");
        fs::set_permissions(&target, fs::Permissions::from_mode(0o600))
            .expect("private target library");
        std::os::unix::fs::symlink("libcudart.so.12.0", cuda.join("libcudart.so.12"))
            .expect("CUDA soname symlink");

        assert_eq!(
            validate_cuda_library_directory(&cuda, test_policy(&cuda))
                .expect("trusted CUDA directory"),
            cuda.canonicalize().expect("canonical CUDA directory")
        );
    }

    #[test]
    fn cuda_library_tree_accepts_4096_entries_and_rejects_4097() {
        let cuda = tempdir().expect("CUDA directory");
        make_private_directory(cuda.path());
        let policy = test_policy(cuda.path());

        for index in 0..CUDA_LIBRARY_TREE_MAX_ENTRIES {
            let library = cuda.path().join(format!("library-{index:04}.so"));
            fs::write(&library, b"").expect("synthetic CUDA library");
            fs::set_permissions(&library, fs::Permissions::from_mode(0o600))
                .expect("private synthetic CUDA library");
        }

        assert_eq!(
            validate_cuda_library_directory(cuda.path(), policy)
                .expect("4096 CUDA tree entries are admitted"),
            cuda.path()
                .canonicalize()
                .expect("canonical CUDA directory")
        );

        let over_limit = cuda
            .path()
            .join(format!("library-{CUDA_LIBRARY_TREE_MAX_ENTRIES:04}.so"));
        fs::write(&over_limit, b"").expect("entry above the CUDA tree limit");
        fs::set_permissions(&over_limit, fs::Permissions::from_mode(0o600))
            .expect("private entry above the CUDA tree limit");

        assert!(validate_cuda_library_directory(cuda.path(), policy).is_err());
    }

    #[test]
    fn cuda_path_accepts_40_symlinks_and_rejects_41() {
        let root = tempdir().expect("CUDA test root");
        make_private_directory(root.path());
        let cuda = root.path().join("cuda");
        fs::create_dir(&cuda).expect("CUDA directory");
        make_private_directory(&cuda);
        let target = cuda.join("libcudart.so.12.0");
        fs::write(&target, b"synthetic library").expect("target library");
        fs::set_permissions(&target, fs::Permissions::from_mode(0o600))
            .expect("private target library");

        let mut target_name = OsString::from("libcudart.so.12.0");
        for index in (0..CUDA_PATH_MAX_SYMLINKS).rev() {
            let link_name = OsString::from(format!("libcudart-link-{index:02}"));
            std::os::unix::fs::symlink(&target_name, cuda.join(&link_name))
                .expect("CUDA symlink chain");
            target_name = link_name;
        }
        let policy = test_policy(&cuda);

        assert_eq!(
            secure_resolve(&cuda.join(&target_name), ExpectedPathKind::File, policy)
                .expect("40 CUDA symlink traversals are admitted"),
            target
        );
        validate_cuda_library_directory(&cuda, policy)
            .expect("CUDA tree containing a 40-link chain is admitted");

        let over_limit = cuda.join("libcudart-link-over-limit");
        std::os::unix::fs::symlink(&target_name, &over_limit)
            .expect("CUDA symlink above the traversal limit");

        assert!(secure_resolve(&over_limit, ExpectedPathKind::File, policy).is_err());
        assert!(validate_cuda_library_directory(&cuda, policy).is_err());
    }

    #[test]
    fn root_owned_sticky_directory_is_only_allowed_for_traversal() {
        let directory = tempdir().expect("sticky traversal directory");
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o1777))
            .expect("root-owned sticky directory mode");
        let metadata = fs::symlink_metadata(directory.path()).expect("sticky directory metadata");
        let policy = CudaPathPolicy {
            user_uid: if metadata.uid() == 0 { 1 } else { 0 },
            root_uid: metadata.uid(),
        };

        validate_directory(&metadata, true, policy)
            .expect("root-owned sticky traversal ancestor is admitted");
        assert!(validate_directory(&metadata, false, policy).is_err());
    }

    #[test]
    fn sidecar_command_removes_dynamic_loader_environment() {
        let mut command = Command::new("/bin/true");
        let loader_names = [
            OsString::from("LD_PRELOAD"),
            OsString::from("LD_PROFILE_OUTPUT"),
            OsString::from_vec(b"LD_NON_UTF8_\xff".to_vec()),
        ];
        for name in &loader_names {
            command.env(name, "attacker-controlled");
        }
        command.env(GLIBC_TUNABLES, "attacker-controlled");
        command.env("TRANSLATOR_SAFE_MARKER", "retained");

        remove_dynamic_loader_environment(&mut command, loader_names.clone());

        let configured = command
            .as_std()
            .get_envs()
            .collect::<std::collections::HashMap<_, _>>();
        for name in &loader_names {
            assert_eq!(configured.get(name.as_os_str()), Some(&None));
        }
        assert_eq!(configured.get(OsStr::new(GLIBC_TUNABLES)), Some(&None));
        assert_eq!(
            configured.get(OsStr::new("TRANSLATOR_SAFE_MARKER")),
            Some(&Some(OsStr::new("retained")))
        );
    }
}
