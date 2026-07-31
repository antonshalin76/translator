use std::fmt;
use std::fs::{self, File};
use std::io::Write;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};

use fs4::{FileExt, TryLockError};
use rustix::fs::{AtFlags, CWD, Mode, OFlags, mkdirat, openat, renameat, unlinkat};
use rustix::io::Errno;
use subtle::ConstantTimeEq;
use thiserror::Error;

const RUNTIME_DIRECTORY: &str = "translator";
const LOCK_FILE: &str = "control.lock";
const TOKEN_FILE: &str = "control.token";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecureRuntimeErrorCode {
    UnsafePath,
    DaemonAlreadyRunning,
    RandomnessUnavailable,
    Io,
}

impl SecureRuntimeErrorCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UnsafePath => "unsafe_runtime_path",
            Self::DaemonAlreadyRunning => "daemon_already_running",
            Self::RandomnessUnavailable => "randomness_unavailable",
            Self::Io => "secure_runtime_io",
        }
    }
}

#[derive(Debug, Error)]
#[error("{code}")]
pub struct SecureRuntimeError {
    code: SecureRuntimeErrorCode,
}

impl SecureRuntimeError {
    pub(crate) const fn new(code: SecureRuntimeErrorCode) -> Self {
        Self { code }
    }

    pub const fn code(&self) -> SecureRuntimeErrorCode {
        self.code
    }
}

impl fmt::Display for SecureRuntimeErrorCode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone)]
pub struct ControlToken {
    encoded: [u8; 64],
}

impl ControlToken {
    pub fn parse(value: &str) -> Result<Self, SecureRuntimeError> {
        let bytes = value.as_bytes();
        if bytes.len() != 64
            || !bytes
                .iter()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(SecureRuntimeError::new(SecureRuntimeErrorCode::UnsafePath));
        }
        let mut encoded = [0_u8; 64];
        encoded.copy_from_slice(bytes);
        Ok(Self { encoded })
    }

    pub(crate) fn matches(&self, candidate: &str) -> bool {
        let candidate = candidate.as_bytes();
        candidate.len() == self.encoded.len() && bool::from(self.encoded.ct_eq(candidate))
    }
}

impl fmt::Debug for ControlToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ControlToken([REDACTED])")
    }
}

pub struct RuntimeLease {
    runtime_dir: File,
    lock: File,
    token_path: PathBuf,
}

impl RuntimeLease {
    pub fn acquire(runtime_parent: &Path) -> Result<Self, SecureRuntimeError> {
        let parent = open_directory(CWD, runtime_parent)?;
        let parent_metadata = parent
            .metadata()
            .map_err(|_| SecureRuntimeError::new(SecureRuntimeErrorCode::Io))?;
        let owner = parent_metadata.uid();

        match mkdirat(
            &parent,
            RUNTIME_DIRECTORY,
            Mode::RUSR | Mode::WUSR | Mode::XUSR,
        ) {
            Ok(()) | Err(Errno::EXIST) => {}
            Err(_) => return Err(SecureRuntimeError::new(SecureRuntimeErrorCode::Io)),
        }
        let runtime_dir = open_directory(&parent, RUNTIME_DIRECTORY)?;
        secure_directory(&runtime_dir, owner)?;

        let lock = open_regular(
            &runtime_dir,
            LOCK_FILE,
            OFlags::RDWR | OFlags::CREATE | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            owner,
        )?;
        match FileExt::try_lock(&lock) {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                return Err(SecureRuntimeError::new(
                    SecureRuntimeErrorCode::DaemonAlreadyRunning,
                ));
            }
            Err(TryLockError::Error(_)) => {
                return Err(SecureRuntimeError::new(SecureRuntimeErrorCode::Io));
            }
        }

        validate_existing_token(&runtime_dir, owner)?;
        rotate_token(&runtime_dir, owner)?;

        Ok(Self {
            runtime_dir,
            lock,
            token_path: runtime_parent.join(RUNTIME_DIRECTORY).join(TOKEN_FILE),
        })
    }

    pub fn token_path(&self) -> &Path {
        &self.token_path
    }
}

impl fmt::Debug for RuntimeLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeLease")
            .field("runtime_dir", &"[OPEN]")
            .field("lock", &"[HELD]")
            .field("token_path", &self.token_path)
            .finish()
    }
}

impl Drop for RuntimeLease {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.lock);
        let _ = self.runtime_dir.sync_all();
    }
}

pub(crate) fn open_directory<Fd: std::os::fd::AsFd>(
    parent: Fd,
    path: impl rustix::path::Arg,
) -> Result<File, SecureRuntimeError> {
    let fd = openat(
        parent,
        path,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|_| SecureRuntimeError::new(SecureRuntimeErrorCode::UnsafePath))?;
    let file = File::from(fd);
    if !file
        .metadata()
        .map_err(|_| SecureRuntimeError::new(SecureRuntimeErrorCode::Io))?
        .is_dir()
    {
        return Err(SecureRuntimeError::new(SecureRuntimeErrorCode::UnsafePath));
    }
    Ok(file)
}

pub(crate) fn secure_directory(directory: &File, owner: u32) -> Result<(), SecureRuntimeError> {
    let metadata = directory
        .metadata()
        .map_err(|_| SecureRuntimeError::new(SecureRuntimeErrorCode::Io))?;
    if !metadata.is_dir() || metadata.uid() != owner {
        return Err(SecureRuntimeError::new(SecureRuntimeErrorCode::UnsafePath));
    }
    directory
        .set_permissions(fs::Permissions::from_mode(0o700))
        .map_err(|_| SecureRuntimeError::new(SecureRuntimeErrorCode::Io))
}

pub(crate) fn open_regular(
    parent: &File,
    name: &str,
    flags: OFlags,
    owner: u32,
) -> Result<File, SecureRuntimeError> {
    let fd = openat(parent, name, flags, Mode::RUSR | Mode::WUSR)
        .map_err(|_| SecureRuntimeError::new(SecureRuntimeErrorCode::UnsafePath))?;
    let file = File::from(fd);
    let metadata = file
        .metadata()
        .map_err(|_| SecureRuntimeError::new(SecureRuntimeErrorCode::Io))?;
    if !metadata.is_file() || metadata.uid() != owner || metadata.nlink() != 1 {
        return Err(SecureRuntimeError::new(SecureRuntimeErrorCode::UnsafePath));
    }
    file.set_permissions(fs::Permissions::from_mode(0o600))
        .map_err(|_| SecureRuntimeError::new(SecureRuntimeErrorCode::Io))?;
    Ok(file)
}

fn validate_existing_token(parent: &File, owner: u32) -> Result<(), SecureRuntimeError> {
    match openat(
        parent,
        TOKEN_FILE,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    ) {
        Ok(fd) => {
            let file = File::from(fd);
            let metadata = file
                .metadata()
                .map_err(|_| SecureRuntimeError::new(SecureRuntimeErrorCode::Io))?;
            if !metadata.is_file() || metadata.uid() != owner || metadata.nlink() != 1 {
                return Err(SecureRuntimeError::new(SecureRuntimeErrorCode::UnsafePath));
            }
            Ok(())
        }
        Err(Errno::NOENT) => Ok(()),
        Err(_) => Err(SecureRuntimeError::new(SecureRuntimeErrorCode::UnsafePath)),
    }
}

fn rotate_token(parent: &File, owner: u32) -> Result<(), SecureRuntimeError> {
    let mut random = [0_u8; 32];
    getrandom::fill(&mut random)
        .map_err(|_| SecureRuntimeError::new(SecureRuntimeErrorCode::RandomnessUnavailable))?;
    let mut encoded = [0_u8; 64];
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for (index, byte) in random.iter().copied().enumerate() {
        encoded[index * 2] = HEX[usize::from(byte >> 4)];
        encoded[index * 2 + 1] = HEX[usize::from(byte & 0x0f)];
    }

    let mut suffix = [0_u8; 8];
    getrandom::fill(&mut suffix)
        .map_err(|_| SecureRuntimeError::new(SecureRuntimeErrorCode::RandomnessUnavailable))?;
    let temporary = format!(
        ".control.token.{}",
        suffix
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );

    let result = (|| {
        let mut file = open_regular(
            parent,
            &temporary,
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            owner,
        )?;
        file.write_all(&encoded)
            .map_err(|_| SecureRuntimeError::new(SecureRuntimeErrorCode::Io))?;
        file.sync_all()
            .map_err(|_| SecureRuntimeError::new(SecureRuntimeErrorCode::Io))?;
        renameat(parent, &temporary, parent, TOKEN_FILE)
            .map_err(|_| SecureRuntimeError::new(SecureRuntimeErrorCode::Io))?;
        parent
            .sync_all()
            .map_err(|_| SecureRuntimeError::new(SecureRuntimeErrorCode::Io))
    })();
    if result.is_err() {
        let _ = unlinkat(parent, &temporary, AtFlags::empty());
    }
    result
}
