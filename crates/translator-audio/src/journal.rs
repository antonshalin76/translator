use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use fs4::FileExt;
use rustix::fs::{AtFlags, CWD, Mode, OFlags, openat, renameat, unlinkat};
use rustix::io::Errno;
use serde::{Deserialize, Serialize};

use crate::{AudioGraphError, AudioGraphErrorCode, EndpointRole};

const JOURNAL_SCHEMA_VERSION: u8 = 1;
static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct OwnedModule {
    pub role: EndpointRole,
    pub module_id: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct OwnershipJournal {
    pub schema_version: u8,
    pub generation: String,
    pub modules: Vec<OwnedModule>,
}

impl OwnershipJournal {
    pub fn empty(generation: String) -> Self {
        Self {
            schema_version: JOURNAL_SCHEMA_VERSION,
            generation,
            modules: Vec::new(),
        }
    }

    pub fn module_for(&self, role: EndpointRole) -> Option<u32> {
        self.modules
            .iter()
            .find(|module| module.role == role)
            .map(|module| module.module_id)
    }

    pub fn module_ids(&self) -> Vec<u32> {
        self.modules.iter().map(|module| module.module_id).collect()
    }

    pub fn validate(&self) -> Result<(), AudioGraphError> {
        if self.schema_version != JOURNAL_SCHEMA_VERSION
            || !valid_generation(&self.generation)
            || self.modules.len() > 3
        {
            return Err(AudioGraphError::new(
                AudioGraphErrorCode::OwnershipJournalInvalid,
            ));
        }
        let mut previous_role_index = None;
        for (index, module) in self.modules.iter().enumerate() {
            let role_index = EndpointRole::ORDER
                .iter()
                .position(|role| *role == module.role)
                .ok_or_else(|| {
                    AudioGraphError::new(AudioGraphErrorCode::OwnershipJournalInvalid)
                })?;
            if previous_role_index.is_some_and(|previous| previous >= role_index)
                || self.modules[..index]
                    .iter()
                    .any(|existing| existing.module_id == module.module_id)
            {
                return Err(AudioGraphError::new(
                    AudioGraphErrorCode::OwnershipJournalInvalid,
                ));
            }
            previous_role_index = Some(role_index);
        }
        Ok(())
    }
}

fn valid_generation(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
}

#[derive(Debug)]
pub(crate) struct JournalStore {
    path: PathBuf,
}

impl JournalStore {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn lock(&self) -> Result<JournalSession, AudioGraphError> {
        let parent = self
            .path
            .parent()
            .ok_or_else(|| AudioGraphError::new(AudioGraphErrorCode::OwnershipJournalInvalid))?;
        fs::create_dir_all(parent)
            .map_err(|_| AudioGraphError::new(AudioGraphErrorCode::OwnershipJournalIo))?;
        let parent_fd = openat(
            CWD,
            parent,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(|_| AudioGraphError::new(AudioGraphErrorCode::OwnershipJournalInvalid))?;
        let parent_file = File::from(parent_fd);
        let metadata = parent_file
            .metadata()
            .map_err(|_| AudioGraphError::new(AudioGraphErrorCode::OwnershipJournalIo))?;
        if !metadata.is_dir() {
            return Err(AudioGraphError::new(
                AudioGraphErrorCode::OwnershipJournalInvalid,
            ));
        }
        parent_file
            .set_permissions(fs::Permissions::from_mode(0o700))
            .map_err(|_| AudioGraphError::new(AudioGraphErrorCode::OwnershipJournalIo))?;

        let journal_name =
            self.path.file_name().map(OsString::from).ok_or_else(|| {
                AudioGraphError::new(AudioGraphErrorCode::OwnershipJournalInvalid)
            })?;
        let lock_name = OsString::from(format!(".{}.lock", journal_name.to_string_lossy()));
        let lock_fd = openat(
            &parent_file,
            &lock_name,
            OFlags::RDWR | OFlags::CREATE | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::RUSR | Mode::WUSR,
        )
        .map_err(|_| AudioGraphError::new(AudioGraphErrorCode::OwnershipJournalInvalid))?;
        let lock_file = File::from(lock_fd);
        if !lock_file
            .metadata()
            .map_err(|_| AudioGraphError::new(AudioGraphErrorCode::OwnershipJournalIo))?
            .is_file()
        {
            return Err(AudioGraphError::new(
                AudioGraphErrorCode::OwnershipJournalInvalid,
            ));
        }
        lock_file
            .set_permissions(fs::Permissions::from_mode(0o600))
            .map_err(|_| AudioGraphError::new(AudioGraphErrorCode::OwnershipJournalIo))?;
        FileExt::lock(&lock_file)
            .map_err(|_| AudioGraphError::new(AudioGraphErrorCode::OwnershipJournalIo))?;
        Ok(JournalSession {
            parent: parent_file,
            _lock: lock_file,
            journal_name,
        })
    }
}

pub(crate) struct JournalSession {
    parent: File,
    _lock: File,
    journal_name: OsString,
}

impl JournalSession {
    pub fn load(&self) -> Result<Option<OwnershipJournal>, AudioGraphError> {
        let journal_fd = match openat(
            &self.parent,
            &self.journal_name,
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        ) {
            Ok(file) => file,
            Err(Errno::NOENT) => return Ok(None),
            Err(_) => {
                return Err(AudioGraphError::new(
                    AudioGraphErrorCode::OwnershipJournalInvalid,
                ));
            }
        };
        let mut file = File::from(journal_fd);
        if !file
            .metadata()
            .map_err(|_| AudioGraphError::new(AudioGraphErrorCode::OwnershipJournalIo))?
            .is_file()
        {
            return Err(AudioGraphError::new(
                AudioGraphErrorCode::OwnershipJournalInvalid,
            ));
        }
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|_| AudioGraphError::new(AudioGraphErrorCode::OwnershipJournalIo))?;
        let journal: OwnershipJournal = serde_json::from_slice(&bytes)
            .map_err(|_| AudioGraphError::new(AudioGraphErrorCode::OwnershipJournalInvalid))?;
        journal.validate()?;
        Ok(Some(journal))
    }

    pub fn save(&self, journal: &OwnershipJournal) -> Result<(), AudioGraphError> {
        journal.validate()?;
        let temporary = OsString::from(format!(
            ".{}.{}.{}.tmp",
            self.journal_name.to_string_lossy(),
            std::process::id(),
            TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let bytes = serde_json::to_vec(journal)
            .map_err(|_| AudioGraphError::new(AudioGraphErrorCode::OwnershipJournalInvalid))?;
        let write_result = (|| {
            let file_fd = openat(
                &self.parent,
                &temporary,
                OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::RUSR | Mode::WUSR,
            )?;
            let mut file = File::from(file_fd);
            file.write_all(&bytes)?;
            file.set_permissions(fs::Permissions::from_mode(0o600))?;
            file.sync_all()?;
            renameat(&self.parent, &temporary, &self.parent, &self.journal_name)?;
            self.parent.sync_all()
        })();
        if write_result.is_err() {
            let _ = unlinkat(&self.parent, &temporary, AtFlags::empty());
            return Err(AudioGraphError::new(
                AudioGraphErrorCode::OwnershipJournalIo,
            ));
        }
        Ok(())
    }

    pub fn remove(&self) -> Result<(), AudioGraphError> {
        match unlinkat(&self.parent, &self.journal_name, AtFlags::empty()) {
            Ok(()) => Ok(()),
            Err(Errno::NOENT) => Ok(()),
            Err(_) => Err(AudioGraphError::new(
                AudioGraphErrorCode::OwnershipJournalIo,
            )),
        }
    }
}
