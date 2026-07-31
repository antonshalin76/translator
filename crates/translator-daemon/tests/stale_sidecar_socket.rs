use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::os::unix::net::UnixListener;

use tempfile::tempdir;
use translator_daemon::{
    ChildState, StaleSocketError, VerifiedStaleSocket, remove_stale_sidecar_socket,
};

fn secure_temp() -> tempfile::TempDir {
    let temp = tempdir().unwrap();
    std::fs::set_permissions(temp.path(), PermissionsExt::from_mode(0o700)).unwrap();
    temp
}

#[test]
fn missing_socket_is_already_clean_only_after_reap() {
    let temp = secure_temp();
    let socket = temp.path().join("sidecar.sock");
    let uid = std::fs::symlink_metadata(temp.path()).unwrap().uid();
    assert_eq!(
        remove_stale_sidecar_socket(&socket, uid, ChildState::Running).unwrap_err(),
        StaleSocketError::ChildStillRunning
    );
    remove_stale_sidecar_socket(&socket, uid, ChildState::Reaped).unwrap();
}

#[test]
fn live_socket_is_never_removed() {
    let temp = secure_temp();
    let socket = temp.path().join("sidecar.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    let uid = std::fs::symlink_metadata(&socket).unwrap().uid();
    assert_eq!(
        remove_stale_sidecar_socket(&socket, uid, ChildState::Reaped).unwrap_err(),
        StaleSocketError::LiveSocket
    );
    assert!(socket.exists());
    drop(listener);
}

#[test]
fn refused_owned_socket_is_removed_after_reap() {
    let temp = secure_temp();
    let socket = temp.path().join("sidecar.sock");
    drop(UnixListener::bind(&socket).unwrap());
    let uid = std::fs::symlink_metadata(&socket).unwrap().uid();
    remove_stale_sidecar_socket(&socket, uid, ChildState::Reaped).unwrap();
    assert!(!socket.exists());
}

#[test]
fn unsafe_socket_and_parent_inodes_fail_closed() {
    let temp = secure_temp();
    let uid = std::fs::symlink_metadata(temp.path()).unwrap().uid();
    let socket = temp.path().join("sidecar.sock");
    std::os::unix::fs::symlink(temp.path().join("missing"), &socket).unwrap();
    assert_eq!(
        remove_stale_sidecar_socket(&socket, uid, ChildState::Reaped).unwrap_err(),
        StaleSocketError::UnsafeInode
    );
    std::fs::remove_file(&socket).unwrap();

    std::fs::write(&socket, b"foreign").unwrap();
    assert_eq!(
        remove_stale_sidecar_socket(&socket, uid, ChildState::Reaped).unwrap_err(),
        StaleSocketError::UnsafeInode
    );
    std::fs::remove_file(&socket).unwrap();

    drop(UnixListener::bind(&socket).unwrap());
    assert_eq!(
        remove_stale_sidecar_socket(&socket, uid + 1, ChildState::Reaped).unwrap_err(),
        StaleSocketError::ForeignOwner
    );
    std::fs::remove_file(&socket).unwrap();

    let linked_parent = temp.path().join("linked");
    let real_parent = temp.path().join("real");
    std::fs::create_dir(&real_parent).unwrap();
    std::fs::set_permissions(&real_parent, PermissionsExt::from_mode(0o700)).unwrap();
    std::os::unix::fs::symlink(&real_parent, &linked_parent).unwrap();
    assert_eq!(
        remove_stale_sidecar_socket(&linked_parent.join("sidecar.sock"), uid, ChildState::Reaped,)
            .unwrap_err(),
        StaleSocketError::InsecureParent
    );

    let parent_file = temp.path().join("parent-file");
    std::fs::write(&parent_file, b"file").unwrap();
    assert_eq!(
        remove_stale_sidecar_socket(&parent_file.join("sidecar.sock"), uid, ChildState::Reaped,)
            .unwrap_err(),
        StaleSocketError::InsecureParent
    );

    let insecure_parent = temp.path().join("insecure");
    std::fs::create_dir(&insecure_parent).unwrap();
    std::fs::set_permissions(&insecure_parent, PermissionsExt::from_mode(0o755)).unwrap();
    assert_eq!(
        remove_stale_sidecar_socket(
            &insecure_parent.join("sidecar.sock"),
            uid,
            ChildState::Reaped,
        )
        .unwrap_err(),
        StaleSocketError::InsecureParent
    );
}

#[test]
fn inode_replacement_between_verify_and_remove_survives() {
    let temp = secure_temp();
    let socket = temp.path().join("sidecar.sock");
    drop(UnixListener::bind(&socket).unwrap());
    let uid = std::fs::symlink_metadata(&socket).unwrap().uid();
    let verified = VerifiedStaleSocket::verify(&socket, uid, ChildState::Reaped).unwrap();

    std::fs::remove_file(&socket).unwrap();
    std::fs::write(&socket, b"replacement").unwrap();
    assert_eq!(
        verified.remove().unwrap_err(),
        StaleSocketError::InodeChanged
    );
    assert_eq!(std::fs::read(&socket).unwrap(), b"replacement");
}

#[test]
fn replacement_created_after_quarantine_survives_exact_stale_inode_removal() {
    let temp = secure_temp();
    let socket = temp.path().join("sidecar.sock");
    drop(UnixListener::bind(&socket).unwrap());
    let original = std::fs::symlink_metadata(&socket).unwrap();
    let uid = original.uid();
    let quarantined = VerifiedStaleSocket::verify(&socket, uid, ChildState::Reaped)
        .unwrap()
        .quarantine()
        .unwrap();
    assert!(!socket.exists());
    let quarantine_entries = std::fs::read_dir(temp.path())
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(quarantine_entries.len(), 1);
    let quarantine_metadata = quarantine_entries[0].metadata().unwrap();
    assert_eq!(quarantine_metadata.dev(), original.dev());
    assert_eq!(quarantine_metadata.ino(), original.ino());

    std::fs::write(&socket, b"new-generation-socket-placeholder").unwrap();
    quarantined.remove().unwrap();

    assert_eq!(
        std::fs::read(&socket).unwrap(),
        b"new-generation-socket-placeholder"
    );
    assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 1);
}
