use std::fs;
use std::os::unix::fs::{MetadataExt, PermissionsExt, symlink};

use translator_daemon::{RuntimeLease, SecureRuntimeErrorCode};

#[test]
fn token_rotates_and_runtime_permissions_are_user_only() {
    let temp = tempfile::tempdir().unwrap();

    let first = RuntimeLease::acquire(temp.path()).unwrap();
    let first_value = fs::read_to_string(first.token_path()).unwrap();
    drop(first);
    let second = RuntimeLease::acquire(temp.path()).unwrap();
    let second_value = fs::read_to_string(second.token_path()).unwrap();

    assert_ne!(first_value, second_value);
    assert_eq!(first_value.len(), 64);
    assert_eq!(second_value.len(), 64);
    assert!(
        first_value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    );
    assert!(
        second_value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    );
    assert_eq!(
        fs::metadata(temp.path().join("translator"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o700
    );
    let metadata = fs::metadata(second.token_path()).unwrap();
    assert!(metadata.is_file());
    assert_eq!(metadata.permissions().mode() & 0o777, 0o600);
    assert_eq!(metadata.uid(), fs::metadata(temp.path()).unwrap().uid());
    let lock_metadata = fs::metadata(temp.path().join("translator/control.lock")).unwrap();
    assert!(lock_metadata.is_file());
    assert_eq!(lock_metadata.permissions().mode() & 0o777, 0o600);
    assert_eq!(
        lock_metadata.uid(),
        fs::metadata(temp.path()).unwrap().uid()
    );
}

#[test]
fn lock_symlink_is_rejected_without_touching_target() {
    let temp = tempfile::tempdir().unwrap();
    let runtime_dir = temp.path().join("translator");
    fs::create_dir(&runtime_dir).unwrap();
    fs::set_permissions(&runtime_dir, fs::Permissions::from_mode(0o700)).unwrap();
    let target = temp.path().join("private-lock-target");
    fs::write(&target, b"private-lock-marker").unwrap();
    symlink(&target, runtime_dir.join("control.lock")).unwrap();

    let error = RuntimeLease::acquire(temp.path()).unwrap_err();

    assert_eq!(error.code(), SecureRuntimeErrorCode::UnsafePath);
    assert_eq!(fs::read(&target).unwrap(), b"private-lock-marker");
}

#[test]
fn token_symlink_is_rejected_without_touching_target() {
    let temp = tempfile::tempdir().unwrap();
    let runtime_dir = temp.path().join("translator");
    fs::create_dir(&runtime_dir).unwrap();
    fs::set_permissions(&runtime_dir, fs::Permissions::from_mode(0o700)).unwrap();
    let target = temp.path().join("private-target");
    fs::write(&target, b"private-target-marker").unwrap();
    symlink(&target, runtime_dir.join("control.token")).unwrap();

    let error = RuntimeLease::acquire(temp.path()).unwrap_err();

    assert_eq!(error.code(), SecureRuntimeErrorCode::UnsafePath);
    assert_eq!(fs::read(&target).unwrap(), b"private-target-marker");
}

#[test]
fn runtime_parent_symlink_is_rejected_without_touching_target() {
    let temp = tempfile::tempdir().unwrap();
    let target = tempfile::tempdir().unwrap();
    symlink(target.path(), temp.path().join("translator")).unwrap();

    let error = RuntimeLease::acquire(temp.path()).unwrap_err();

    assert_eq!(error.code(), SecureRuntimeErrorCode::UnsafePath);
    assert!(fs::read_dir(target.path()).unwrap().next().is_none());
}

#[test]
fn concurrent_lease_fails_without_rotating_token_then_succeeds_after_drop() {
    let temp = tempfile::tempdir().unwrap();
    let first = RuntimeLease::acquire(temp.path()).unwrap();
    let token_before = fs::read_to_string(first.token_path()).unwrap();

    let error = RuntimeLease::acquire(temp.path()).unwrap_err();

    assert_eq!(error.code(), SecureRuntimeErrorCode::DaemonAlreadyRunning);
    assert_eq!(
        fs::read_to_string(first.token_path()).unwrap(),
        token_before
    );

    drop(first);
    let second = RuntimeLease::acquire(temp.path()).unwrap();
    assert_ne!(
        fs::read_to_string(second.token_path()).unwrap(),
        token_before
    );
}
