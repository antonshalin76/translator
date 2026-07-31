use std::collections::VecDeque;
use std::fmt;
use std::fs::File;
use std::io::{self, Write};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use rustix::fs::{Mode, OFlags, mkdirat};
use rustix::io::Errno;
use serde::Serialize;

use crate::secure_state::{open_directory, open_regular, secure_directory};
use crate::{SecureRuntimeError, SecureRuntimeErrorCode};

const MAX_DEBUG_TEXT_EVENTS: usize = 200;
const MAX_DEBUG_TEXT_BYTES: usize = 1024 * 1024;
const DEFAULT_MAX_DURATION_MS: u64 = 10 * 60 * 1_000;
const DEFAULT_MAX_BYTES: u64 = 500 * 1024 * 1024;
const DEFAULT_MINIMUM_FREE_BYTES: u64 = 5 * 1024 * 1024 * 1024;

#[derive(Clone)]
pub struct DebugTextEvent {
    transcript: String,
    translation: String,
}

impl DebugTextEvent {
    pub fn new(transcript: impl Into<String>, translation: impl Into<String>) -> Self {
        Self {
            transcript: transcript.into(),
            translation: translation.into(),
        }
    }

    fn bytes(&self) -> usize {
        self.transcript.len() + self.translation.len()
    }
}

impl fmt::Debug for DebugTextEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("DebugTextEvent([REDACTED])")
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct DebugTextStatus {
    pub enabled: bool,
    pub event_count: usize,
    pub bytes_used: usize,
}

#[derive(Default)]
pub struct DebugTextBuffer {
    enabled: bool,
    events: VecDeque<DebugTextEvent>,
    bytes_used: usize,
}

impl fmt::Debug for DebugTextBuffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DebugTextBuffer")
            .field("enabled", &self.enabled)
            .field("event_count", &self.events.len())
            .field("bytes_used", &self.bytes_used)
            .finish()
    }
}

impl DebugTextBuffer {
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.clear();
        }
    }

    pub fn push(&mut self, event: DebugTextEvent) -> bool {
        if !self.enabled {
            return false;
        }
        let event_bytes = event.bytes();
        if event_bytes > MAX_DEBUG_TEXT_BYTES {
            return false;
        }
        while self.events.len() >= MAX_DEBUG_TEXT_EVENTS
            || self.bytes_used + event_bytes > MAX_DEBUG_TEXT_BYTES
        {
            let Some(removed) = self.events.pop_front() else {
                break;
            };
            self.bytes_used -= removed.bytes();
        }
        self.bytes_used += event_bytes;
        self.events.push_back(event);
        true
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn bytes_used(&self) -> usize {
        self.bytes_used
    }

    pub fn safe_status(&self) -> DebugTextStatus {
        DebugTextStatus {
            enabled: self.enabled,
            event_count: self.events.len(),
            bytes_used: self.bytes_used,
        }
    }

    pub fn clear_for_provider_switch(&mut self) {
        self.clear();
    }

    pub fn clear_for_session_stop(&mut self) {
        self.clear();
    }

    fn clear(&mut self) {
        self.events.clear();
        self.bytes_used = 0;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DebugCaptureLimits {
    max_duration_ms: u64,
    max_bytes: u64,
    minimum_free_bytes: u64,
}

impl Default for DebugCaptureLimits {
    fn default() -> Self {
        Self {
            max_duration_ms: DEFAULT_MAX_DURATION_MS,
            max_bytes: DEFAULT_MAX_BYTES,
            minimum_free_bytes: DEFAULT_MINIMUM_FREE_BYTES,
        }
    }
}

impl DebugCaptureLimits {
    pub const fn new(max_duration_ms: u64, max_bytes: u64, minimum_free_bytes: u64) -> Self {
        Self {
            max_duration_ms,
            max_bytes,
            minimum_free_bytes,
        }
    }

    pub const fn max_duration_ms(self) -> u64 {
        self.max_duration_ms
    }

    pub const fn max_bytes(self) -> u64 {
        self.max_bytes
    }

    pub const fn minimum_free_bytes(self) -> u64 {
        self.minimum_free_bytes
    }
}

pub trait FreeSpaceProbe: Send + Sync + fmt::Debug {
    fn available_bytes(&self, directory: &File) -> io::Result<u64>;
}

#[derive(Debug)]
struct FilesystemFreeSpace;

impl FreeSpaceProbe for FilesystemFreeSpace {
    fn available_bytes(&self, directory: &File) -> io::Result<u64> {
        let stats = rustix::fs::fstatvfs(directory)?;
        Ok(stats.f_frsize.saturating_mul(stats.f_bavail))
    }
}

#[derive(Debug)]
pub struct DebugCaptureStore {
    directory: Arc<File>,
    directory_path: PathBuf,
    limits: DebugCaptureLimits,
    probe: Arc<dyn FreeSpaceProbe>,
    active: Arc<AtomicBool>,
}

impl DebugCaptureStore {
    pub fn open(
        state_parent: &Path,
        limits: DebugCaptureLimits,
    ) -> Result<Self, SecureRuntimeError> {
        Self::open_with_probe(state_parent, limits, Arc::new(FilesystemFreeSpace))
    }

    pub fn open_with_probe(
        state_parent: &Path,
        limits: DebugCaptureLimits,
        probe: Arc<dyn FreeSpaceProbe>,
    ) -> Result<Self, SecureRuntimeError> {
        let parent = open_directory(rustix::fs::CWD, state_parent)?;
        let owner = parent
            .metadata()
            .map_err(|_| SecureRuntimeError::new(SecureRuntimeErrorCode::Io))?
            .uid();
        let translator = open_or_create_directory(&parent, "translator", owner)?;
        let directory = open_or_create_directory(&translator, "debug", owner)?;
        Ok(Self {
            directory: Arc::new(directory),
            directory_path: state_parent.join("translator/debug"),
            limits,
            probe,
            active: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn start(
        &self,
        capture_name: &str,
        started_at_ms: u64,
    ) -> Result<DebugCaptureSession, SecureRuntimeError> {
        if !valid_capture_name(capture_name)
            || self
                .active
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return Err(SecureRuntimeError::new(SecureRuntimeErrorCode::UnsafePath));
        }
        let result = (|| {
            let available = self
                .probe
                .available_bytes(&self.directory)
                .map_err(|_| SecureRuntimeError::new(SecureRuntimeErrorCode::Io))?;
            if available < self.limits.minimum_free_bytes {
                return Err(SecureRuntimeError::new(SecureRuntimeErrorCode::Io));
            }
            let file_name = format!("{capture_name}.pcm");
            let file = open_regular(
                &self.directory,
                &file_name,
                OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                self.directory
                    .metadata()
                    .map_err(|_| SecureRuntimeError::new(SecureRuntimeErrorCode::Io))?
                    .uid(),
            )?;
            Ok(DebugCaptureSession {
                file,
                path: self.directory_path.join(file_name),
                directory: Arc::clone(&self.directory),
                limits: self.limits,
                probe: Arc::clone(&self.probe),
                active: Arc::clone(&self.active),
                started_at_ms,
                bytes_written: 0,
                stop_reason: None,
            })
        })();
        if result.is_err() {
            self.active.store(false, Ordering::Release);
        }
        result
    }

    pub fn directory_path(&self) -> &Path {
        &self.directory_path
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugCaptureStopReason {
    TimeLimit,
    ByteLimit,
    LowFreeSpace,
    Io,
}

pub struct DebugCaptureSession {
    file: File,
    path: PathBuf,
    directory: Arc<File>,
    limits: DebugCaptureLimits,
    probe: Arc<dyn FreeSpaceProbe>,
    active: Arc<AtomicBool>,
    started_at_ms: u64,
    bytes_written: u64,
    stop_reason: Option<DebugCaptureStopReason>,
}

impl fmt::Debug for DebugCaptureSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DebugCaptureSession")
            .field("path", &self.path)
            .field("bytes_written", &self.bytes_written)
            .field("stop_reason", &self.stop_reason)
            .finish()
    }
}

impl DebugCaptureSession {
    pub fn append(&mut self, bytes: &[u8], at_ms: u64) -> Result<(), DebugCaptureStopReason> {
        self.expire(at_ms)?;
        let write_len =
            u64::try_from(bytes.len()).map_err(|_| DebugCaptureStopReason::ByteLimit)?;
        if self.bytes_written.saturating_add(write_len) > self.limits.max_bytes {
            return self.stop(DebugCaptureStopReason::ByteLimit);
        }
        let available = self.probe.available_bytes(&self.directory);
        let available = match available {
            Ok(available) => available,
            Err(_) => return self.stop(DebugCaptureStopReason::Io),
        };
        if available < self.limits.minimum_free_bytes.saturating_add(write_len) {
            return self.stop(DebugCaptureStopReason::LowFreeSpace);
        }
        if self.file.write_all(bytes).is_err() {
            return self.stop(DebugCaptureStopReason::Io);
        }
        self.bytes_written += write_len;
        Ok(())
    }

    pub fn deadline_ms(&self) -> u64 {
        self.started_at_ms
            .saturating_add(self.limits.max_duration_ms)
    }

    pub fn expire(&mut self, at_ms: u64) -> Result<(), DebugCaptureStopReason> {
        if let Some(reason) = self.stop_reason {
            return Err(reason);
        }
        if at_ms >= self.deadline_ms() {
            return self.stop(DebugCaptureStopReason::TimeLimit);
        }
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn stop(&mut self, reason: DebugCaptureStopReason) -> Result<(), DebugCaptureStopReason> {
        self.stop_reason = Some(reason);
        self.active.store(false, Ordering::Release);
        let _ = self.file.sync_all();
        Err(reason)
    }
}

impl Drop for DebugCaptureSession {
    fn drop(&mut self) {
        self.active.store(false, Ordering::Release);
        let _ = self.file.sync_all();
    }
}

fn open_or_create_directory(
    parent: &File,
    name: &str,
    owner: u32,
) -> Result<File, SecureRuntimeError> {
    match mkdirat(parent, name, Mode::RUSR | Mode::WUSR | Mode::XUSR) {
        Ok(()) | Err(Errno::EXIST) => {}
        Err(_) => return Err(SecureRuntimeError::new(SecureRuntimeErrorCode::Io)),
    }
    let directory = open_directory(parent, name)?;
    secure_directory(&directory, owner)?;
    Ok(directory)
}

fn valid_capture_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}
