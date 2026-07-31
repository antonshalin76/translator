use std::{
    ffi::OsString,
    fs::File,
    os::{
        fd::AsRawFd,
        unix::fs::{MetadataExt, PermissionsExt},
    },
    path::{Path, PathBuf},
};

use rustix::{
    fs::{
        AtFlags, CWD, FileType, Mode, OFlags, RenameFlags, openat, renameat_with, statat, unlinkat,
    },
    io::Errno,
    net::{AddressFamily, SocketAddrUnix, SocketFlags, SocketType, connect, socket_with},
};
use thiserror::Error;

use crate::ChildState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum StaleSocketError {
    #[error("sidecar child is still running")]
    ChildStillRunning,
    #[error("sidecar socket still accepts connections")]
    LiveSocket,
    #[error("sidecar socket inode is unsafe")]
    UnsafeInode,
    #[error("sidecar socket has an unexpected owner")]
    ForeignOwner,
    #[error("sidecar socket parent is insecure")]
    InsecureParent,
    #[error("sidecar socket inode changed after verification")]
    InodeChanged,
    #[error("sidecar socket operation failed")]
    Io,
}

#[derive(Debug)]
pub struct VerifiedStaleSocket {
    parent: File,
    name: OsString,
    device: u64,
    inode: u64,
}

#[derive(Debug)]
pub struct QuarantinedStaleSocket {
    parent: File,
    name: OsString,
    device: u64,
    inode: u64,
}

impl VerifiedStaleSocket {
    pub fn verify(
        path: &Path,
        expected_uid: u32,
        child_state: ChildState,
    ) -> Result<Self, StaleSocketError> {
        match verify_optional(path, expected_uid, child_state)? {
            Some(verified) => Ok(verified),
            None => Err(StaleSocketError::InodeChanged),
        }
    }

    pub fn remove(self) -> Result<(), StaleSocketError> {
        self.quarantine()?.remove()
    }

    pub fn quarantine(self) -> Result<QuarantinedStaleSocket, StaleSocketError> {
        let current =
            statat(&self.parent, &self.name, AtFlags::SYMLINK_NOFOLLOW).map_err(|error| {
                if error == Errno::NOENT {
                    StaleSocketError::InodeChanged
                } else {
                    StaleSocketError::Io
                }
            })?;
        if current.st_dev != self.device
            || current.st_ino != self.inode
            || !FileType::from_raw_mode(current.st_mode).is_socket()
        {
            return Err(StaleSocketError::InodeChanged);
        }

        let quarantine_name = quarantine_name()?;
        renameat_with(
            &self.parent,
            &self.name,
            &self.parent,
            &quarantine_name,
            RenameFlags::NOREPLACE,
        )
        .map_err(|_| StaleSocketError::InodeChanged)?;
        let quarantined = statat(&self.parent, &quarantine_name, AtFlags::SYMLINK_NOFOLLOW)
            .map_err(|_| StaleSocketError::InodeChanged)?;
        if quarantined.st_dev != self.device
            || quarantined.st_ino != self.inode
            || !FileType::from_raw_mode(quarantined.st_mode).is_socket()
        {
            let _ = renameat_with(
                &self.parent,
                &quarantine_name,
                &self.parent,
                &self.name,
                RenameFlags::NOREPLACE,
            );
            return Err(StaleSocketError::InodeChanged);
        }
        self.parent.sync_all().map_err(|_| StaleSocketError::Io)?;
        Ok(QuarantinedStaleSocket {
            parent: self.parent,
            name: quarantine_name,
            device: self.device,
            inode: self.inode,
        })
    }
}

impl QuarantinedStaleSocket {
    pub fn remove(self) -> Result<(), StaleSocketError> {
        let current =
            statat(&self.parent, &self.name, AtFlags::SYMLINK_NOFOLLOW).map_err(|error| {
                if error == Errno::NOENT {
                    StaleSocketError::InodeChanged
                } else {
                    StaleSocketError::Io
                }
            })?;
        if current.st_dev != self.device
            || current.st_ino != self.inode
            || !FileType::from_raw_mode(current.st_mode).is_socket()
        {
            return Err(StaleSocketError::InodeChanged);
        }
        unlinkat(&self.parent, &self.name, AtFlags::empty()).map_err(|error| {
            if error == Errno::NOENT {
                StaleSocketError::InodeChanged
            } else {
                StaleSocketError::Io
            }
        })?;
        self.parent.sync_all().map_err(|_| StaleSocketError::Io)?;
        Ok(())
    }
}

pub fn remove_stale_sidecar_socket(
    path: &Path,
    expected_uid: u32,
    child_state: ChildState,
) -> Result<(), StaleSocketError> {
    if let Some(verified) = verify_optional(path, expected_uid, child_state)? {
        verified.remove()?;
    }
    Ok(())
}

fn verify_optional(
    path: &Path,
    expected_uid: u32,
    child_state: ChildState,
) -> Result<Option<VerifiedStaleSocket>, StaleSocketError> {
    if child_state != ChildState::Reaped {
        return Err(StaleSocketError::ChildStillRunning);
    }
    let parent_path = path.parent().ok_or(StaleSocketError::InsecureParent)?;
    let name = path
        .file_name()
        .map(OsString::from)
        .ok_or(StaleSocketError::UnsafeInode)?;
    let parent_fd = openat(
        CWD,
        parent_path,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|_| StaleSocketError::InsecureParent)?;
    let parent = File::from(parent_fd);
    let parent_metadata = parent
        .metadata()
        .map_err(|_| StaleSocketError::InsecureParent)?;
    if !parent_metadata.is_dir() || parent_metadata.permissions().mode() & 0o777 != 0o700 {
        return Err(StaleSocketError::InsecureParent);
    }

    let socket = match statat(&parent, &name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(socket) => socket,
        Err(Errno::NOENT) => return Ok(None),
        Err(_) => return Err(StaleSocketError::Io),
    };
    if !FileType::from_raw_mode(socket.st_mode).is_socket() || socket.st_nlink != 1 {
        return Err(StaleSocketError::UnsafeInode);
    }
    if socket.st_uid != expected_uid {
        return Err(StaleSocketError::ForeignOwner);
    }
    if parent_metadata.uid() != expected_uid {
        return Err(StaleSocketError::InsecureParent);
    }

    let descriptor_path =
        PathBuf::from(format!("/proc/self/fd/{}", parent.as_raw_fd())).join(&name);
    let probe = socket_with(
        AddressFamily::UNIX,
        SocketType::STREAM,
        SocketFlags::NONBLOCK | SocketFlags::CLOEXEC,
        None,
    )
    .map_err(|_| StaleSocketError::Io)?;
    let address = SocketAddrUnix::new(descriptor_path).map_err(|_| StaleSocketError::Io)?;
    match connect(&probe, &address) {
        Ok(()) => return Err(StaleSocketError::LiveSocket),
        Err(Errno::CONNREFUSED) => {}
        Err(Errno::NOENT) => {
            return Err(StaleSocketError::InodeChanged);
        }
        Err(_) => return Err(StaleSocketError::LiveSocket),
    }

    Ok(Some(VerifiedStaleSocket {
        parent,
        name,
        device: socket.st_dev,
        inode: socket.st_ino,
    }))
}

fn quarantine_name() -> Result<OsString, StaleSocketError> {
    let mut random = [0_u8; 16];
    getrandom::fill(&mut random).map_err(|_| StaleSocketError::Io)?;
    let suffix = random
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(OsString::from(format!(".sidecar.sock.stale-{suffix}")))
}
