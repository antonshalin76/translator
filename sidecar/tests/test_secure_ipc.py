import os
from pathlib import Path

import pytest

from translator_sidecar.secure_ipc import open_private_directory


def test_private_parent_remains_pinned_after_path_replacement(tmp_path: Path):
    original = tmp_path / "original"
    original.mkdir(mode=0o700)
    descriptor = open_private_directory(original)
    try:
        moved = tmp_path / "moved"
        original.rename(moved)
        original.mkdir(mode=0o700)
        assert os.fstat(descriptor).st_ino == moved.stat().st_ino
        assert os.fstat(descriptor).st_ino != original.stat().st_ino
        assert not os.get_inheritable(descriptor)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    "failure", ["ancestor_link", "leaf_link", "file", "mode", "missing"]
)
def test_invalid_private_path_fails_without_descriptor_leak(
    tmp_path: Path, failure: str
):
    parent = tmp_path / "real"
    parent.mkdir(mode=0o700)
    target = parent / "session"
    target.mkdir(mode=0o700)
    if failure == "ancestor_link":
        (tmp_path / "alias").symlink_to(parent, target_is_directory=True)
        target = tmp_path / "alias/session"
    elif failure == "leaf_link":
        (tmp_path / "alias").symlink_to(target, target_is_directory=True)
        target = tmp_path / "alias"
    elif failure == "file":
        target = parent / "file"
        target.write_bytes(b"not a directory")
    elif failure == "mode":
        target.chmod(0o755)
    else:
        target = parent / "missing"
    before = len(os.listdir("/proc/self/fd"))
    with pytest.raises(OSError):
        open_private_directory(target)
    assert len(os.listdir("/proc/self/fd")) == before
